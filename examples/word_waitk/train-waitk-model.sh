#!/bin/bash

DATAPATH=$1
MAXTOKENS=$2
UPDATEFREQ=$3
SRC=$4
TGT=$5
WAITK=$6
UNIT=$7
NUM_NODES=$8
NODE_RANK=$9
MASTER_IP="${10}"
MASTER_PORT="${11}"

has_no_space="jv km lo my su th"
if [ `echo $TGT | grep -c "zh" ` -gt 0 ]; then
TOKENIZER="zh"
elif [ `echo $TGT | grep -c "ja" ` -gt 0 ]; then
TOKENIZER="ja-mecab"
elif [[ " $has_no_space " =~ .*\ $TGT\ .* ]]; then
TOKENIZER="char"
else
TOKENIZER="intl"
fi

if [[ -z $WAITK ]]; then
    WAITK=9
fi

if [[ $UNIT == "word" ]]; then
    SAVEDIR="${DATAPATH}/checkpoints/word_waitk_beam1_k${WAITK}"
    EXTRA_ARGS="--word-waitk"
else
    SAVEDIR="${DATAPATH}/checkpoints/waitk_beam1_k${WAITK}"
    EXTRA_ARGS=""
fi

MKL_THREADING_LAYER=GNU torchrun --nproc_per_node=8 \
	--nnodes=${NUM_NODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_IP} \
	--master_port=${MASTER_PORT} $(which fairseq-train) $DATAPATH --arch waitk_transformer --task waitk_translation \
	--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 0.0005 --stop-min-lr '1e-09' \
	--lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
	--dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy \
	--label-smoothing 0.1 --max-tokens $MAXTOKENS --save-dir $SAVEDIR \
	--tensorboard-logdir $SAVEDIR --fp16 --update-freq $UPDATEFREQ --max-update 1200000 \
	--save-interval 1 --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
	--eval-bleu --eval-bleu-remove-bpe sentencepiece --max-tokens-valid 2000 \
	--eval-bleu-args '{"beam": 1, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok=space \
	--source-lang $SRC --target-lang $TGT --sacrebleu-tokenizer $TOKENIZER --keep-last-epochs 5 \
	--multi-waitk --eval-waitk $WAITK $EXTRA_ARGS --min-waitk 1 --share-all-embeddings
