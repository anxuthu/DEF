lr=0.4
p=128
seed=0
e=300
url="tcp://localhost:23450"

for seed in 0
do
    file="vgg16bn-lsgd${lr}-p${p}w8-${seed}"
    echo "$file" | tee $file
    NCCL_IB_DISABLE=1 python main.py -o LSGD --period $p --lr $lr --seed $seed --dist-url $url --rank 0 --world-size 2 --epochs $e | tee -a $file &
    NCCL_IB_DISABLE=1 python main.py -o LSGD --period $p --lr $lr --seed $seed --dist-url $url --rank 1 --world-size 2 --epochs $e | tee -a $file &
    wait
done
