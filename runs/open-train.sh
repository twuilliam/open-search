cd ..

DATA_DIR=data/domainnet
EXP_DIR=exp/domainnet

##
# domainnet many-shot
##
for DOMAIN in clipart infograph painting real sketch
do
    python train.py \
        --epochs=60 \
        --multi-gpu \
        --data_dir=$DATA_DIR \
        --exp_dir=$EXP_DIR/manyshot \
        --dataset=domainnet_$DOMAIN \
        --mode=im \
        --lr=0.001 \
        --da \
        --backbone=seresnet \
        --temperature=0.05
done

python train.py \
    --epochs=60 \
    --multi-gpu \
    --data_dir=$DATA_DIR \
    --exp_dir=$EXP_DIR/manyshot \
    --dataset=domainnet \
    --mode=sk \
    --lr=0.001 \
    --da \
    --backbone=seresnet \
    --temperature=0.05

##
# domainnet zero-shot
##
for DOMAIN in clipart infograph painting real sketch
do
    python train.py \
        --epochs=30 \
        --multi-gpu \
        --data_dir=$DATA_DIR \
        --exp_dir=$EXP_DIR/zeroshot \
        --dataset=domainnet_$DOMAIN \
        --mode=im \
        --lr=0.001 \
        --da \
        --backbone=seresnet \
        --temperature=0.05 \
        --overwrite
done

python train.py \
    --epochs=30 \
    --multi-gpu \
    --data_dir=$DATA_DIR \
    --exp_dir=$EXP_DIR/zeroshot \
    --dataset=domainnet \
    --mode=sk \
    --lr=0.001 \
    --da \
    --backbone=seresnet \
    --temperature=0.05 \
    --overwrite
