arch="resnet50"
lr=0.1
b=32
coeff=0.3
seed=0
e=100
p=1
ratio=64
url="tcp://localhost:23450"
devices="0,1,2,3"

file="${arch}-def${coeff}lr${lr}b${b}-p${p}r${ratio}w8-${seed}"
echo "$file" | tee $file
for rank in 0 1
do
    CUDA_VISIBLE_DEVICES=${devices} python main.py -a $arch -o DEF --coeff $coeff --lr $lr -b $b -wd 1e-4 -p $p --ratio $ratio --seed $seed --dist-url $url --rank $rank --world-size 2 --dist-backend GLOO --path /data/ILSVRC2012 --dataset imagenet --epochs $e -ls const -ds 0.3 0.6 0.9 -wp 0 | tee -a $file &
done
wait
