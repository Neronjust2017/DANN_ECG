#!/usr/bin/env bash

times=4
for j in `seq 1 $times`
do
max=3
for i in `seq 0 $max`
do
python main_baseline.py \
--lr=0.001 \
--logs='run_'$j'/logs_'$i'/' \
--batch_size=128 \
--n_epoch=100 \
--model_root='run_'$j'/models_'$i'/' \
--unseen_index=$i
done
done

# --lr=5e-4 --num_classes=7 --test_every=500 --logs='run_0/logs/' --batch_size=64 --model_path='run_0/models/' --unseen_index=0 --inner_loops=45001 --step_size=15000 --state_dict='' --data_root=''