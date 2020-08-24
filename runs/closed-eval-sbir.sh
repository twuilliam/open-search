cd ..

DATA_DIR=data/sketches
EXP_DIR=exp/sketches

python retrieve.py \
    --sk-path=$EXP_DIR/zeroshot/Sketchy_sk/checkpoint.pth.tar \
    --im-path=$EXP_DIR/zeroshot/Sketchy_im/checkpoint.pth.tar

python retrieve.py \
    --sk-path=$EXP_DIR/zeroshot/TU-Berlin_sk/checkpoint.pth.tar \
    --im-path=$EXP_DIR/zeroshot/TU-Berlin_im/checkpoint.pth.tar

python retrieve.py \
    --sk-path=$EXP_DIR/gzsl/Sketchy_sk/checkpoint.pth.tar \
    --im-path=$EXP_DIR/gzsl/Sketchy_im/checkpoint.pth.tar

python retrieve.py \
    --sk-path=$EXP_DIR/gzsl/TU-Berlin_sk/checkpoint.pth.tar \
    --im-path=$EXP_DIR/gzsl/TU-Berlin_im/checkpoint.pth.tar
