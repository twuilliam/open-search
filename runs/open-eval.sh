cd ..

DATA_DIR=data/domainnet
EXP_DIR=exp/domainnet

##
# Feature extraction
##

# extract many shot
for DOMAIN in clipart infograph painting real sketch
do
    python retrieve.py \
        --im-path=$EXP_DIR/manyshot/domainnet_im_$DOMAIN/checkpoint.pth.tar \
        --sk-path=$EXP_DIR/manyshot/domainnet_sk/checkpoint.pth.tar
done

# extract zero shot
for DOMAIN in clipart infograph painting real sketch
do
    python retrieve.py \
        --im-path=$EXP_DIR/zeroshot/domainnet_im_$DOMAIN/checkpoint.pth.tar \
        --sk-path=$EXP_DIR/zeroshot/domainnet_sk/checkpoint.pth.tar
done

##
# any2any experiment
##

python retrieve-any.py --dir-path=$EXP_DIR/manyshot
python retrieve-any.py --dir-path=$EXP_DIR/zeroshot

##
# many2any experiment
##

python retrieve-many.py --dir-path=$EXP_DIR/zeroshot --eval=many2any
python retrieve-many.py --dir-path=$EXP_DIR/manyshot --eval=many2any

##
# any2many experiment
##

python retrieve-many.py --dir-path=$EXP_DIR/manyshot --eval=any2many
