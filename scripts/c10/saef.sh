lr=5e-2
b=16
seed=0
e=200
ratio=64
p2=0
url="tcp://localhost:23450"
devices="0,1"

file="vgg16bn-saef${lr}b${b}-r${ratio}p${p2}w8-${seed}"
echo "$file" | tee $file
for rank in 0 1 2 3
do
	CUDA_VISIBLE_DEVICES=${devices} python main.py -o SAEF --lr $lr -b $b -p2 $p2 --ratio $ratio --seed $seed --dist-url $url --rank $rank --world-size 4 --epochs $e --dist-backend GLOO --path /data/cifar -ls const -ds 0.5 0.75 -wp 0 | tee -a $file &
done
wait
