#!/bin/bash

node=$1 
hours=$2
shared=$3

if [ "$node" = "v100" ]; then
    partition="V4V32_SKY32M192_L"
elif [ "$node" = "a100" ]; then    
    partition="A2V80_ICE56M256_L"
elif [ "$node" = "p100" ]; then
    partition="P4V12_SKY32M192_L"
else
    echo "Unknown node type: $node"
    exit 1
fi 

if [ "$shared" = "share" ]; then
    echo "salloc -A gol_xli281_uksr -t ${hours}:00:00 -p ${partition} -n 16 --mem=64g  --gres=gpu"
    tmux has-session -t testnode || tmux new -d -s testnode
    tmux send-keys -t testnode "salloc -A gol_xli281_uksr -t ${hours}:00:00 -p ${partition} -n 16 --mem=64g  --gres=gpu" Enter
    sleep 2
    squeue -u $(whoami):
else
    echo "salloc -A gol_xli281_uksr -t ${hours}:00:00 -p ${partition} -N 1 --exclusive --gres=gpu"
    tmux has-session -t testnode || tmux new -d -s testnode
    tmux send-keys -t testnode "salloc -A gol_xli281_uksr -t ${hours}:00:00 -p ${partition} -N 1 --exclusive --gres=gpu" Enter
    sleep 2
    squeue -u $(whoami)
fi


