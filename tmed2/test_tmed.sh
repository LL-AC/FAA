cuda=$1
node=$2
epochs=$3
data_path=$4
batch_size=$5
model=$6
lr=$7
loss=$8
mode=$9
fold=${10}
logger=${11}
config=${12}

cd ../..

echo $fold

CUDA_VISIBLE_DEVICES=$cuda torchrun --nproc_per_node $node --master_port 12346  main.py --epochs $epochs  --dataset tmed2 --data_path $data_path \
 --batch_size $batch_size --accum_iter 1 --log_dir "./logger/"$logger"/"$fold"/log" --output_dir "./logger/"$logger"/"$fold --pin_memory --model $model --lr $lr \
 --warmup_epochs 2 --weight_decay 0.005 --ctiterion $loss --evaluate --resume "./logger/"$logger"/"$fold"/checkpoint-"$mode".pth" --config_path "$config" --num_classes 3