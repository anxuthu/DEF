import numpy as np
import torch

__all__ = ['RankK', 'URSB', 'TopK']

class Reducer:
    def __init__(self, random_seed, device):
        self.rng = np.random.RandomState(random_seed)
        M = 1024 * 1024
        self.precalc_numbers = (
            torch.from_numpy(self.rng.randn(128 * M)).to(device).type(torch.float32)
        )
        if torch.distributed.is_available():
            self.n_workers = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.n_workers = 1
            self.rank = 0
        self.device = device

    def reduce(self, grad_in, grad_out, memory_out):
        """Return communicated bits"""
        raise NotImplementedError()


class TopK(Reducer):
    """
    Use same amount as rank-based
    """
    def __init__(self, random_seed, device, compression=1 / 244, **kwargs):
        super().__init__(random_seed, device)
        self.compression = compression

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        # Find the size of a flatpacked gradient
        flatgrad_size = 0
        tensor_idx = [0]
        for tensor in grad_in:
            top_size = max(1, int(0.5 * self.compression * tensor.nelement()))
            flatgrad_size += top_size
            tensor_idx.append(tensor_idx[-1] + top_size)
        flatgrad_start_idx = tensor_idx[:-1]
        flatgrad_end_idx = tensor_idx[1:]
        flat_values = torch.empty(flatgrad_size, device=self.device)
        flat_positions = torch.empty(flatgrad_size, device=self.device, dtype=torch.int)

        for tensor, start, end in zip(grad_in, flatgrad_start_idx, flatgrad_end_idx):
            top_size = max(1, int(0.5 * self.compression * tensor.nelement()))
            _, positions = torch.topk(tensor.view(-1).abs(), top_size, sorted=False)
            values = tensor.view(-1)[positions].contiguous()
            flat_values[start:end] = values
            flat_positions[start:end] = positions

        for tensor, mem, start, end in zip(
            grad_in, memory_out, flatgrad_start_idx, flatgrad_end_idx
        ):
            positions = flat_positions[start:end]
            mem.data[:] = tensor
            mem.view(-1)[positions.long()] = 0.0

        if self.n_workers > 1:
            worker_values = [torch.empty_like(flat_values) for i in range(self.n_workers)]
            worker_positions = [torch.empty_like(flat_positions) for i in range(self.n_workers)]
            h1 = all_gather(worker_values, flat_values, async_op=True)
            h2 = all_gather(worker_positions, flat_positions, async_op=True)
            h1.wait()
            h2.wait()
        else:
            worker_values = [flat_values]
            worker_positions = [flat_positions]
        bits_communicated += n_bits(flat_values) + n_bits(flat_positions)

        for tensor, out, start, end in zip(
            grad_in, grad_out, flatgrad_start_idx, flatgrad_end_idx
        ):
            out.data[:] = 0
            for pos, val in zip(worker_positions, worker_values):
                positions = pos[start:end]
                values = val[start:end]
                # out.view(-1)[pos].add_(1.0 / self.n_workers, val)
                out.view(-1)[positions.long()] += values / self.n_workers


#class UniformRandomSparseBlockReducer(Reducer):
class URSB(Reducer):
    def __init__(self, random_seed, device, compression=1 / 244, **kwargs):
        super().__init__(random_seed, device)
        self.compression = compression

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        values_list = []
        start_idx_list = []
        block_sizes = []

        for tensor in grad_in:
            block_size = max(1, int(self.compression * tensor.nelement()))
            block_sizes.append(block_size)
            start_idx = self.rng.choice(tensor.nelement())
            start_idx_list.append(start_idx)
            end_idx = min(start_idx + block_size, tensor.nelement())
            bfr = torch.empty(block_size, dtype=torch.float32, device=self.device)
            bfr[: end_idx - start_idx] = tensor.view(-1)[start_idx:end_idx]
            rest = block_size - (end_idx - start_idx)
            if rest > 0:
                bfr[end_idx - start_idx :] = tensor.view(-1)[:rest]
            values_list.append(bfr)

        flat_values = TensorBuffer(values_list)

        for tensor, mem, start_idx, block_size in zip(grad_in, memory_out, start_idx_list, block_sizes):
            end_idx = min(start_idx + block_size, tensor.nelement())
            rest = block_size - (end_idx - start_idx)
            #mem.data = tensor.clone()
            mem.data.copy_(tensor)
            mem.view(-1)[start_idx:end_idx] = 0.0
            if rest > 0:
                mem.view(-1)[:rest] = 0.0

        flat_values.all_reduce()
        flat_values.buffer /= self.n_workers
        bits_communicated += flat_values.bits()

        for tensor, out, start_idx, block_size, values in zip(grad_in, grad_out, start_idx_list, block_sizes, flat_values):
            end_idx = min(start_idx + block_size, tensor.nelement())
            rest = block_size - (end_idx - start_idx)
            out.data.zero_()
            out.view(-1)[start_idx:end_idx] = values[: end_idx - start_idx]
            if rest > 0:
                out.view(-1)[:rest] = values[end_idx - start_idx :]


class RankK(Reducer):
    def __init__(self, random_seed, device, reuse_query=False, rank=1, **kwargs):
        super().__init__(random_seed, device)
        self.rank = rank
        self.p_memory = None
        self.q_memory = None
        self.reuse_query = reuse_query

    def set_random(self, vector):
        torch.manual_seed(self.rng.randint(1_000_000_000))
        vector.data[:] = torch.randn(*vector.shape, device=self.device)
        # orthogonalize(vector)

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        # Split the tensors into rank1-ones that will be reduced un-compressed
        # and rank > 1 tensors that are compressed
        rank1_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() <= 1
        ]
        high_rank_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() > 1
        ]

        # We are building a rank-1 approximation of every tensor
        # that can be interpreted as a matrix. Let the approximation be
        # M = p q^T
        # We are allocating consequtive memory for the p's and q's

        memory_is_uninitialized = self.p_memory is None

        ##### reduce.allocate_memory #####
        p_total_size = 0
        q_total_size = 0
        for tensor, _, _ in high_rank_tensors:
            matrix = tensor.view(tensor.shape[0], -1)
            n, m = matrix.shape
            rank = min(n, m, self.rank)
            p_total_size += n * rank
            q_total_size += m * rank
        if self.p_memory is None:
            self.p_memory = torch.empty(p_total_size, device=self.device)
            self.q_memory = torch.empty(q_total_size, device=self.device)

        # Find them again and make lists of pointers
        ps = []
        qs = []
        p_idx = 0
        q_idx = 0
        for tensor, _, _ in high_rank_tensors:
            matrix = tensor.view(tensor.shape[0], -1)
            n, m = matrix.shape
            rank = min(n, m, self.rank)
            ps.append(self.p_memory[p_idx : p_idx + n * rank].view(n, rank))
            qs.append(self.q_memory[q_idx : q_idx + m * rank].view(m, rank))
            p_idx += n * rank
            q_idx += m * rank

        ##### reduce.prepare.q #####
        for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
            matrix = tensor.view(tensor.shape[0], -1)
            n, m = matrix.shape

            if self.reuse_query and not memory_is_uninitialized:
                # orthogonalize(q)
                pass
            else:
                # Sample a query vector q
                self.set_random(q)

        ##### reduce.compute.p #####
        for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
            matrix = tensor.view(tensor.shape[0], -1)
            torch.matmul(matrix, q, out=p)

        ##### reduce.p #####
        all_reduce(self.p_memory)
        bits_communicated += n_bits(self.p_memory)

        # Start communicating rank 1 tensors
        rank1_tensor_list = TensorBuffer([tensor for (tensor, _, _) in rank1_tensors])
        rank1_handle = rank1_tensor_list.all_reduce(async_op=True)
        bits_communicated += rank1_tensor_list.bits()

        ##### reduce.normalize.p #####
        for p in ps:
            orthogonalize(p)

        ##### reduce.compute.q #####
        for p, q, (tensor, _, mem) in zip(ps, qs, high_rank_tensors):
            matrix = tensor.view(tensor.shape[0], -1)
            torch.matmul(matrix.t(), p, out=q)
            mem.data[:] = tensor - torch.matmul(p, q.t()).view_as(tensor) # 1

        ##### reduce.q #####
        all_reduce(self.q_memory)
        bits_communicated += n_bits(self.q_memory)
        self.q_memory.data[:] /= self.n_workers

        ##### reduce.outerprod #####
        for p, q, (tensor, out, mem) in zip(ps, qs, high_rank_tensors):
            # Set the output gradient
            output = torch.matmul(p, q.t()).view_as(tensor)
            #mem.data[:] = tensor - output # 2
            out.data[:] = output

        ##### reduce.rank1.unpack #####
        rank1_handle.wait()
        rank1_tensor_list.buffer /= self.n_workers
        rank1_tensor_list.unpack([out for (_, out, _) in rank1_tensors])

        #fp_bits = n_bits(torch.cat([tensor.view(-1) for tensor in grad_in]))
        #print(fp_bits / bits_communicated)
        #import sys; sys.exit()


@torch.jit.script
def orthogonalize(matrix):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col


def n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()


class TensorBuffer():
    """
    Packs multiple tensors into one flat buffer for efficient
    intra-worker communication.
    """
    def __init__(self, tensors):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors = tensors

        self.buffer = torch.cat([t.view(-1) for t in tensors]) # copies
    
    def __getitem__(self, index):
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(*self._tensors[index].shape)

    def __len__(self):
        return len(self._tensors)

    def pack(self, tensors=None):
        # Optional. init already does this.
        if tensors is None:
            tensors = self._tensors
        for tensor, entry in zip(tensors, self):
            entry[:] = tensor

    def unpack(self, tensors):
        for tensor, entry in zip(tensors, self):
            tensor[:] = entry

    def nelement(self):
        return self.buffer.nelement()

    def element_size(self):
        return self.buffer.element_size()

    def bits(self):
        return 8 * self.nelement() * self.element_size()

    def all_reduce(self, async_op=False):
        return torch.distributed.all_reduce(self.buffer, async_op=async_op)
    
    def all_gather(self, async_op=False):
        n_workers = torch.distributed.get_world_size() if torch.distributed.is_available() else 1
        buffers = [torch.empty_like(self.buffer) for i in range(n_workers)]
        handle = all_gather(buffers, self.buffer, async_op=async_op)
        if async_op:
            return buffers, handle
        else:
            return buffers


def all_gather(out_list, in_tensor, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_gather(out_list, in_tensor, **kwargs)
    else:
        assert len(out_list) == 1
        out_list[0].data = in_tensor


def all_reduce(*args, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_reduce(*args, **kwargs)
