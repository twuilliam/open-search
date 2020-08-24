cd ..

DATA_DIR=data/sketches
EXP_DIR=exp/sketches

python fewshot.py \
    --sk-path=$EXP_DIR/fewshot/Sketchy_sk/checkpoint.pth.tar \
    --im-path=$EXP_DIR/fewshot/Sketchy_im/checkpoint.pth.tar \
    --mixing
