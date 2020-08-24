cd ..

DATA_DIR=data/shapes
EXP_DIR=exp/shapes

##
# SHREC13
##
python train.py \
    --epochs=100 \
    --multi-gpu \
    --data_dir=$DATA_DIR \
    --exp_dir=$EXP_DIR \
    --dataset=SHREC13 \
    --mode=sk \
    --batch-size=128 \
    --lr=0.01 \
    --da \
    --shape \
    --word=shrec \
    --backbone=seresnet \
    --temperature=0.1

python train.py \
    --epochs=30 \
    --multi-gpu \
    --data_dir=$DATA_DIR \
    --exp_dir=$EXP_DIR \
    --dataset=SHREC13 \
    --mode=im \
    --batch-size=128 \
    --lr=0.01 \
    --shape \
    --word=shrec \
    --backbone=seresnet \
    --temperature=0.1

##
# SHREC14
##
python train.py \
    --epochs=100 \
    --multi-gpu \
    --data_dir=$DATA_DIR \
    --exp_dir=$EXP_DIR \
    --dataset=SHREC14 \
    --mode=sk \
    --batch-size=128 \
    --lr=0.01 \
    --da \
    --shape \
    --word=shrec \
    --backbone=seresnet \
    --temperature=0.1

python train.py \
    --epochs=30 \
    --multi-gpu \
    --data_dir=$DATA_DIR \
    --exp_dir=$EXP_DIR \
    --dataset=SHREC14 \
    --mode=im \
    --batch-size=128 \
    --lr=0.01 \
    --shape \
    --word=shrec \
    --backbone=seresnet \
    --temperature=0.1

##
# PART-SHREC14
##
python train.py \
    --epochs=100 \
    --multi-gpu \
    --data_dir=$DATA_DIR \
    --exp_dir=$EXP_DIR \
    --dataset=PART-SHREC14 \
    --mode=sk \
    --batch-size=128 \
    --lr=0.01 \
    --shape \
    --da \
    --word=shrec \
    --backbone=seresnet \
    --temperature=0.1

python train.py \
    --epochs=30 \
    --multi-gpu \
    --data_dir=$DATA_DIR \
    --exp_dir=$EXP_DIR \
    --dataset=PART-SHREC14 \
    --mode=im \
    --batch-size=128 \
    --lr=0.01 \
    --shape \
    --word=shrec \
    --backbone=seresnet \
    --temperature=0.1
