#!/bin/bash
# experiment with scheduler step
#python PP_MRI_segmentation_main.py --epochs 40 --num_files 4000 --lr 3e-3 --loss "dl" --sch_step 8
#python PP_MRI_segmentation_main.py --epochs 40 --num_files 4000 --lr 3e-3 --loss "dl" --sch_step 10
#python PP_MRI_segmentation_main.py --epochs 40 --num_files 4000 --lr 3e-3 --loss "dl" --sch_step 20

# experiment with learning rate
#python PP_MRI_segmentation_main.py --epochs 40 --num_files 4000 --lr 1e-3 --loss "dl" --sch_step 10
#python PP_MRI_segmentation_main.py --epochs 40 --num_files 4000 --lr 1e-2 --loss "dl" --sch_step 10

# experiment with cost functions
#python PP_MRI_segmentation_main.py --epochs 40 --num_files 4000 --lr 3e-3 --loss "gdl" --sch_step 10
#python PP_MRI_segmentation_main.py --epochs 40 --num_files 4000 --lr 3e-3 --loss "dl+bce" --sch_step 10

# experiment with batch size
#python PP_MRI_segmentation_main.py --epochs 40 --num_files 4000 --lr 3e-3 --loss "dl" --sch_step 10 --train_batch_size 4
#python PP_MRI_segmentation_main.py --epochs 40 --num_files 4000 --lr 3e-3 --loss "dl" --sch_step 10 --train_batch_size 2

# experiment with loss weights
#python PP_MRI_segmentation_main.py --beta 2
#python PP_MRI_segmentation_main.py --beta 3
#python PP_MRI_segmentation_main.py --pw 250 --loss "dl+bce"
#python PP_MRI_segmentation_main.py --pw 125 --loss "dl+bce"
#python PP_MRI_segmentation_main.py --pw 60 --loss "dl+bce"

# experiment with US segmentation models
python PP_US_segmentation.py --epochs 40 --lr 1e-3
python PP_US_segmentation.py --epochs 40 --lr 1e-4
python PP_US_segmentation.py --epochs 40 --lr 1e-5
