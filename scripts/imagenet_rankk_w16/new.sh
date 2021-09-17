rank=$1
arch="resnet50"
lr=0.4
b=128
coeff=0.3
seed=0
e=100
prank=1
url="tcp://10.252.193.56:23450"

file="${arch}-new${coeff}lr${lr}b${b}-r${prank}w16n${rank}-${seed}"
echo "$file" | tee $file
NCCL_IB_DISABLE=1 python main.py --path /export/Data/ILSVRC2012 --dataset imagenet -a $arch -wd 1e-4 -o New --coeff $coeff --lr $lr -b $b -rq --prank $prank --seed $seed --dist-url $url --rank $rank --world-size 4 --epochs $e | tee -a $file &
wait
