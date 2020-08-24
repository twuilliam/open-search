cd ..

DATA_DIR=data/sketches
EXP_DIR=exp/sketches

##
# zero-shot
##

# Sketchy dataset
python train.py \
    --epochs=30 \
    --multi-gpu \
    --data_dir=$DATA_DIR \
    --exp_dir=$EXP_DIR/zeroshot \
    --dataset=Sketchy \
    --mode=sk \
    --lr=0.001 \
    --da \
    --backbone=seresnet \
    --temperature=0.05 \
    --overwrite

python train.py \
    --epochs=10 \
    --multi-gpu \
    --data_dir=$DATA_DIR \
    --exp_dir=$EXP_DIR/zeroshot \
    --dataset=Sketchy \
    --backbone=seresnet \
    --lr=0.001 \
    --da \
    --mode=im \
    --temperature=0.05 \
    --overwrite

# TU-Berlin dataset
python train.py \
    --epochs=50 \
    --multi-gpu \
    --data_dir=$DATA_DIR \
    --exp_dir=$EXP_DIR/zeroshot \
    --dataset=TU-Berlin \
    --mode=sk \
    --lr=0.001 \
    --da \
    --backbone=seresnet \
    --temperature=0.05 \
    --overwrite

python train.py \
    --epochs=5 \
    --multi-gpu \
    --data_dir=$DATA_DIR \
    --exp_dir=$EXP_DIR/zeroshot \
    --dataset=TU-Berlin \
    --backbone=seresnet \
    --lr=0.001 \
    --da \
    --mode=im \
    --temperature=0.05 \
    --overwrite

##
# generalized zero-shot
##

# Sketchy dataset
python train.py \
    --epochs=30 \
    --multi-gpu \
    --data_dir=$DATA_DIR \
    --exp_dir=$EXP_DIR/gzsl \
    --dataset=Sketchy \
    --mode=sk \
    --lr=0.001 \
    --da \
    --temperature=0.05 \
    --overwrite \
    --backbone=vgg16 \
    --gzsl

python train.py \
    --epochs=10 \
    --multi-gpu \
    --data_dir=$DATA_DIR \
    --exp_dir=$EXP_DIR/gzsl \
    --dataset=Sketchy \
    --lr=0.001 \
    --da \
    --mode=im \
    --temperature=0.05 \
    --overwrite \
    --backbone=vgg16 \
    --gzsl

# TU-Berlin dataset
python train.py \
    --epochs=100 \
    --multi-gpu \
    --data_dir=$DATA_DIR \
    --exp_dir=$EXP_DIR/gzsl \
    --dataset=TU-Berlin \
    --mode=sk \
    --lr=0.001 \
    --da \
    --temperature=0.05 \
    --overwrite \
    --backbone=vgg16 \
    --gzsl

python train.py \
    --epochs=20 \
    --multi-gpu \
    --data_dir=$DATA_DIR \
    --exp_dir=$EXP_DIR/gzsl \
    --dataset=TU-Berlin \
    --lr=0.001 \
    --da \
    --mode=im \
    --temperature=0.05 \
    --overwrite \
    --backbone=vgg16 \
    --gzsl
