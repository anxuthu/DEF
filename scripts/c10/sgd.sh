lr=0.4
b=1024
seed=0
e=300
url="tcp://localhost:23450"

for seed in 0
do
    file="vgg16bn-sgd${lr}b${b}-${seed}"
    echo "$file" | tee $file
    python main.py -o SGD --lr $lr -b $b --seed $seed --dist-url $url --rank 0 --world-size 1 --epochs $e --path /data/cifar | tee -a $file
done
