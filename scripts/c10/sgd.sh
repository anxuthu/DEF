lr=5e-2
b=16
seed=0
e=200
url="tcp://localhost:23450"
devices="0,1"

file="vgg16bn-sgd${lr}b${b}w8-${seed}"
echo "$file" | tee $file
for rank in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=${devices} python main.py -o SGD --lr $lr -b $b --seed $seed --dist-url $url --rank $rank --world-size 4 --dist-backend GLOO --path /data/cifar --epochs $e -ls const -ds 0.5 0.75 -wp 0 | tee -a $file &
done
wait
