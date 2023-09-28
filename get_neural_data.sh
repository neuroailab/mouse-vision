#!/bin/bash

base_url=https://mouse-vision-neuraldata.s3.amazonaws.com

for dataset in mouse_neuropixels_visual_data_with_reliabilities mouse_calcium_visual_data_with_reliabilities
do
    mkdir -p ./neural_data
    curl -fLo ./neural_data/${dataset}.pkl ${base_url}/${dataset}.pkl
done
