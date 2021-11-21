lr=5e-2
b=16
seed=2
e=200
ratio=64
p2=0
ratio2=1
url="tcp://localhost:23452"
devices="4,5"

file="vgg16bn-cser-lr${lr}b${b}-r${ratio}p${p2}r${ratio2}w8-${seed}"
echo "$file" | tee $file
for rank in 0 1 2 3
do
	CUDA_VISIBLE_DEVICES=${devices} python main.py -o CSER --lr $lr -b $b -p2 $p2 --ratio $ratio --ratio2 $ratio2 --seed $seed --dist-url $url --rank $rank --world-size 4 --dist-backend GLOO --path /data/cifar --epochs $e -ls const -ds 0.5 0.75 -wp 0 | tee -a $file &
done
wait
