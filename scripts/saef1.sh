lr=0.2
seed=0
e=300
prank=1
url="tcp://10.252.193.56:23450"

for seed in 0
do
    file="vgg16bn-saef${lr}-r${prank}w16n1-${seed}"
    echo "$file" | tee $file
    NCCL_IB_DISABLE=1 python main.py -o SAEF -rq --lr $lr --prank $prank --seed $seed --dist-url $url --rank 2 --world-size 4 --epochs $e | tee -a $file &
    NCCL_IB_DISABLE=1 python main.py -o SAEF -rq --lr $lr --prank $prank --seed $seed --dist-url $url --rank 3 --world-size 4 --epochs $e | tee -a $file &
    wait
done
