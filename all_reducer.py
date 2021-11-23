import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

__all__ = ['URSB', 'GRSB']

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


#class UniformRandomSparseBlockReducer(Reducer):
class URSB(Reducer):
    def __init__(self, random_seed, device, compression=1 / 244, **kwargs):
        super().__init__(random_seed, device)
        self.compression = compression

    def reduce(self, grad_in, grad_out, memory_out):
        if self.compression == 0:
            for t1, t2, mem in zip(grad_in, grad_out, memory_out):
                t2.zero_()
                mem.copy_(t1)
            return
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


#class GlobalRandomSparseBlockReducer(Reducer):
class GRSB(Reducer):
    def __init__(self, random_seed, device, compression=1 / 244, **kwargs):
        super().__init__(random_seed, device)
        self.compression = compression

    def reduce(self, grad_in, grad_out, memory_out):
        if self.compression == 0:
            for t1, t2, mem in zip(grad_in, grad_out, memory_out):
                t2.zero_()
                mem.copy_(t1)
            return
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        values_list = []
        idx_list = []

        r = int(1 / self.compression)
        r1 = 2 ** int(np.log2(r) / 2)
        r2 = r // r1

        for tensor, mem in zip(grad_in, memory_out):
            if len(tensor.shape) > 1:
                size1 = tensor.shape[0]
                k1 = max(1, round(size1 / r1))
                begin1 = self.rng.choice(int(np.ceil(size1 / k1))) * k1
                end1 = min(begin1 + k1, size1)

                size2 = tensor.shape[1]
                k2 = max(1, round(size2 / r2))
                begin2 = self.rng.choice(int(np.ceil(size2 / k2))) * k2
                end2 = min(begin2 + k2, size2)

                value = tensor[begin1:end1, begin2:end2].reshape(-1)
                values_list.append(value)
                idx_list.append((begin1, end1, begin2, end2))

                mem.copy_(tensor)
                mem[begin1:end1, begin2:end2].zero_()
            else:
                size = tensor.shape[0]
                k = max(1, round(size / r))
                begin = self.rng.choice(int(np.ceil(size / k))) * k
                end = min(begin + k, size)

                value = tensor[begin:end]
                values_list.append(value)
                idx_list.append((begin, end))

                mem.copy_(tensor)
                mem[begin:end].zero_()

        flat_values = TensorBuffer(values_list)
        flat_values.all_reduce()
        flat_values.buffer /= self.n_workers
        bits_communicated += flat_values.bits()

        for tensor, out, value, idx in zip(grad_in, grad_out, flat_values, idx_list):
            if len(tensor.shape) > 1:
                begin1, end1, begin2, end2 = idx
                shape = tensor[begin1:end1, begin2:end2].shape
                out.zero_()
                out[begin1:end1, begin2:end2] = value.view(shape)
            else:
                begin, end = idx
                out.zero_()
                out[begin:end] = value

#class GRSB(Reducer):
#    def __init__(self, random_seed, device, compression=1 / 244, **kwargs):
#        super().__init__(random_seed, device)
#        self.compression = compression
#
#    def reduce(self, grad_in, grad_out, memory_out):
#        if self.compression == 0:
#            for t1, t2, mem in zip(grad_in, grad_out, memory_out):
#                t2.zero_()
#                mem.copy_(t1)
#            return
#        """
#        Reduce gradients between the workers in place
#        :param grad_in: dictionary
#        :param grad_out: dictionary
#        :param memory_out: dictionary
#        """
#        grad_in_vec = parameters_to_vector(grad_in)
#        num_p = grad_in_vec.numel()
#
#        total_blocks = 1024 #TODO
#        block_size = int(np.ceil(num_p / total_blocks))
#        vec = torch.zeros(total_blocks, block_size, device=self.device)
#        vec.view(-1)[:num_p] = grad_in_vec
#
#        n_blocks = max(1, int(total_blocks * self.compression))
#        block_idx = self.rng.choice(total_blocks, n_blocks, replace=False)
#
#        grad_out_vec = vec[block_idx, :].clone()
#
#        vec[block_idx, :] = 0
#        vector_to_parameters(vec.view(-1)[:num_p], memory_out)
#
#        all_reduce(grad_out_vec)
#        grad_out_vec /= self.n_workers
#        vec = torch.zeros(total_blocks, block_size, device=self.device)
#        vec[block_idx, :] = grad_out_vec
#        vector_to_parameters(vec.view(-1)[:num_p], grad_out)


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
