lr=0.8
seed=0
e=300
url="tcp://10.252.193.56:23450"

for seed in 0
do
    file="vgg16bn-sgd${lr}-w16n0-${seed}"
    echo "$file" | tee $file
    NCCL_IB_DISABLE=1 python main.py -o SGD --lr $lr --seed $seed --dist-url $url --rank 0 --world-size 4 --epochs $e | tee -a $file &
    NCCL_IB_DISABLE=1 python main.py -o SGD --lr $lr --seed $seed --dist-url $url --rank 1 --world-size 4 --epochs $e | tee -a $file &
    wait
done
