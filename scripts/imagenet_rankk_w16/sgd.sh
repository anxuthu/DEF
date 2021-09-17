rank=$1
arch="resnet50"
lr=0.8
b=128
seed=0
e=100
url="tcp://10.252.193.56:23450"

file="${arch}-sgd${lr}b${b}-w16n${rank}-${seed}"
echo "$file" | tee $file
NCCL_IB_DISABLE=1 python main.py --path /export/Data/ILSVRC2012 --dataset imagenet -a $arch -wd 1e-4 -o SGD --lr $lr -b $b --seed $seed --dist-url $url --rank $rank --world-size 4 --epochs $e | tee -a $file &
wait
