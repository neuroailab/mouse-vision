#!/bin/bash

base_url=https://mouse-vision-ckpts.s3.amazonaws.com

for model in alexnet_bn_ir dual_stream_ir six_stream_simclr shi_mousenet_ir shi_mousenet_vispor5_ir
do
    mkdir -p ./model_ckpts
    curl -fLo ./model_ckpts/${model}.pt ${base_url}/${model}.pt
done
