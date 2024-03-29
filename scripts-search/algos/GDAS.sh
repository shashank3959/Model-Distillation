#!/bin/bash
# bash ./scripts-search/algos/GDAS.sh cifar10 -1

echo script name: $0
echo $# arguments passed

if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need only 2 parameters for dataset and seed"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME environment variable for data dir saving"
  TORCH_HOME="./data"
  echo "Default TORCH_HOME: $TORCH_HOME"
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

dataset=$1
seed=$2
channel=16
num_cells=5
max_nodes=4

if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
  data_path="$TORCH_HOME/cifar.python"
else
  data_path="$TORCH_HOME/cifar.python/ImageNet16"
fi

teacher_path="$TORCH_HOME"

save_dir=./output/cell-search-tiny/GDAS-${dataset}

# tau_max and tau_min are the max and min temperatures in softmax. The temperature is annealed from max to min.

OMP_NUM_THREADS=4 python ./exps/algos/GDAS.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} --data_path ${data_path} \
	--teacher_path ${teacher_path} \
	--search_space_name aa-nas \
	--tau_max 10 --tau_min 0.1 \
	--arch_learning_rate 0.0003 --arch_weight_decay 0.001 \
	--workers 4 --print_freq 200 --rand_seed ${seed}