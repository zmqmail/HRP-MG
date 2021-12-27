#!/bin/bash
ratio=$1

echo Cut data ...
python cut_data.py --ratio $ratio

echo Generate homogeneous ...
python generate_homogeneous.py --ratio $ratio

echo Train model ...
python train.py --ratio $ratio