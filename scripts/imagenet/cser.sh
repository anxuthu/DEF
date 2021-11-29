arch="resnet50"
lr=0.1
b=32
seed=0
e=100
ratio=0
p2=1
ratio2=64
url="tcp://localhost:23450"
devices="0,1,2,3"

file="${arch}-cser-lr${lr}b${b}-r${ratio}p${p2}r${ratio2}w8-${seed}"
echo "$file" | tee $file
for rank in 0 1
do
    CUDA_VISIBLE_DEVICES=${devices} python main.py -a $arch -o CSER --lr $lr -b $b -wd 1e-4 -p2 $p2 --ratio $ratio --ratio2 $ratio2 --seed $seed --dist-url $url --rank $rank --world-size 2 --dist-backend GLOO --path /data/ILSVRC2012 --dataset imagenet --epochs $e -ls const -ds 0.3 0.6 0.9 -wp 0 | tee -a $file &
done
wait
