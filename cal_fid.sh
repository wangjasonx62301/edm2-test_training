#!/bin/bash

CUDA_VISIBLE_DEVICES=3

expr_name=00007-edm2-cifar10
# 23068
for kimg in 06291
do
    (python reconstruct_phema.py --indir=./.training-runs/${expr_name} \
     --outdir=.phema/${expr_name} --outstd=0.045 --batch=8 --outkimg=${kimg})
    for ema_rate in 0.045
    do
        for steps in 64
        do
            (python generate_images.py --net=.phema/${expr_name}/phema-00${kimg}-${ema_rate}.pkl --outdir=.outK/${expr_name}/phema-00${kimg}-${ema_rate}/${steps}s --subdirs --seeds=0-49999 --steps=${steps})
            (python calculate_metrics.py calc --images=.outK/${expr_name}/phema-00${kimg}-${ema_rate}/${steps}s \
             --ref=/home/jasonx62301/for_python/edm2-test_training/.fid-refs/cifar10.pkl \
             --batch=256 --num=50000 --metrics=fid)
        done
    done
done
