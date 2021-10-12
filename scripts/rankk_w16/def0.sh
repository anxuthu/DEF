lr=0.2
b=64
coeff=0.3
seed=0
e=300
prank=1
p=1
p2=0
ratio=1
url="tcp://10.252.193.56:23450"

for seed in 0
do
    file="vgg16bn-def${coeff}lr${lr}b${b}-p${p}r${prank}p${p2}r${ratio}w16n0-${seed}"
    echo "$file" | tee $file
    NCCL_IB_DISABLE=1 python main.py -o DEF --coeff $coeff --lr $lr -b $b -p $p -rq --prank $prank -p2 $p2 --ratio $ratio --seed $seed --dist-url $url --rank 0 --world-size 4 --epochs $e | tee -a $file &
    NCCL_IB_DISABLE=1 python main.py -o DEF --coeff $coeff --lr $lr -b $b -p $p -rq --prank $prank -p2 $p2 --ratio $ratio --seed $seed --dist-url $url --rank 1 --world-size 4 --epochs $e | tee -a $file &
    wait
done
