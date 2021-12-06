arch='vgg16bn'
lr=5e-2
b=16
seed=0
e=200
ratio=0
p2=1
ratio2=64
url="tcp://localhost:23450"
devices="0,1"

file="${arch}-cser-lr${lr}b${b}-r${ratio}p${p2}r${ratio2}w8-${seed}"
echo "$file" | tee $file
for rank in 0 1 2 3
do
	CUDA_VISIBLE_DEVICES=${devices} python main.py -a $arch -o CSER --lr $lr -b $b -p2 $p2 --ratio $ratio --ratio2 $ratio2 --seed $seed --dist-url $url --rank $rank --world-size 4 --dist-backend GLOO --path /data/cifar --epochs $e -ls const -ds 0.5 0.75 -wp 0 | tee -a $file &
done
wait
