cd ..

DATA_DIR=data/sketches
EXP_DIR=exp/sketches

python train.py \
    --epochs=20 \
    --multi-gpu \
    --data_dir=$DATA_DIR \
    --exp_dir=$EXP_DIR/fewshot \
    --dataset=Sketchy \
    --mode=sk \
    --lr=0.001 \
    --da \
    --temperature=0.05 \
    --overwrite \
    --backbone=vgg19 \
    --fewshot

python train.py \
    --epochs=20 \
    --multi-gpu \
    --data_dir=$DATA_DIR \
    --exp_dir=$EXP_DIR/fewshot \
    --dataset=Sketchy \
    --lr=0.001 \
    --da \
    --mode=im \
    --temperature=0.05 \
    --overwrite \
    --backbone=vgg19 \
    --fewshot
