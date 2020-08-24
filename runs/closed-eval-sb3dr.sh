cd ..

DATA_DIR=data/shapes
EXP_DIR=exp/shapes

python retrieve.py \
    --im-path=$EXP_DIR/SHREC13_im/checkpoint.pth.tar \
    --sk-path=$EXP_DIR/SHREC13_sk/checkpoint.pth.tar

python retrieve.py \
    --im-path=$EXP_DIR/SHREC14_im/checkpoint.pth.tar \
    --sk-path=$EXP_DIR/SHREC14_sk/checkpoint.pth.tar

python retrieve.py \
    --im-path=$EXP_DIR/PART-SHREC14_im/checkpoint.pth.tar \
    --sk-path=$EXP_DIR/PART-SHREC14_sk/checkpoint.pth.tar
