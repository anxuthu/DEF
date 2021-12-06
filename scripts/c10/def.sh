arch='vgg16bn'
lr=5e-2
b=16
coeff=0.3
seed=0
e=200
p=1
ratio=64
url="tcp://localhost:23450"
devices="0,1"

file="${arch}-def${coeff}lr${lr}b${b}-p${p}r${ratio}w8-${seed}"
echo "$file" | tee $file
for rank in 0 1 2 3
do
	CUDA_VISIBLE_DEVICES=${devices} python main.py -a $arch -o DEF --coeff $coeff --lr $lr -b $b -p $p --ratio $ratio --seed $seed --dist-url $url --rank $rank --world-size 4 --dist-backend GLOO --path /data/cifar --epochs $e -ls const -ds 0.5 0.75 -wp 0 | tee -a $file &
done
wait
