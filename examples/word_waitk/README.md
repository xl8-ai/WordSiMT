## Word-level Wait-k

### Training

#### Data Processing
Download and extract the data and use the following command.  
```shell
./prepare-data-joint-dict.sh $DATAPATH $SRC $TGT $BPESIZE $DESTDIR
```

#### Train word-level wait-k model:
Run `train-waitk-model.sh`.  
`DATAPATH` should be `${DESTDIR}/${SRC}.${TGT}.${BPESIZE}.joint`, which is the  result of `prepare-data-joint-dict.sh`.  
`WAITK` is the wait-k value used during validation.
```shell
#!/bin/bash

DATAPATH=$1
MAXTOKENS=$2
UPDATEFREQ=$3
SRC=$4
TGT=$5
WAITK=$6
UNIT=$7  # token or word

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

fairseq-train $DATAPATH --arch waitk_transformer --task waitk_translation \
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
```
#### Train word-level wait-k + LM model:
Run `train-waitk-lm-model.sh`.  
It will first copy the word-level wait-k model from `"${DATAPATH}/checkpoints/word_waitk_beam1_k${WAITK}/checkpoint_best.pt"`  
`LANGUAGE_MODEL_NAME` should be the name of the language model in huggingface. In our paper, it is either `facebook/xglm-564M` or `gpt2`.
```shell
#!/bin/bash

DATAPATH=$1
MAXTOKENS=$2
UPDATEFREQ=$3
SRC=$4
TGT=$5
WAITK=$6
UNIT=$7  # token or word
LANGUAGE_MODEL_NAME="${8}"

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
    SAVEDIR="${DATAPATH}/checkpoints/word_waitk_beam1_k${WAITK}_lm"
    if [[ ! -f "${SAVEDIR}/checkpoint_last.pt" ]]; then
        mkdir $SAVEDIR
	    cp "${DATAPATH}/checkpoints/word_waitk_beam1_k${WAITK}/checkpoint_best.pt" "${SAVEDIR}/checkpoint_nmt.pt"
        EXTRA_ARGS="--word-waitk --warmup-from-nmt --reset-lr-scheduler --reset-meters"
    else
        EXTRA_ARGS="--word-waitk"
    fi
else
    SAVEDIR="${DATAPATH}/checkpoints/waitk_beam1_k${WAITK}_lm"
    if [[ ! -f "${SAVEDIR}/checkpoint_last.pt" ]]; then
        mkdir $SAVEDIR
        cp "${DATAPATH}/checkpoints/waitk_beam1_k${WAITK}/checkpoint_best.pt" "${SAVEDIR}/checkpoint_nmt.pt"
        EXTRA_ARGS="--warmup-from-nmt --reset-lr-scheduler --reset-meters"
    else
        EXTRA_ARGS=""
    fi
fi

fairseq-train $DATAPATH --arch waitk_transformer --task waitk_translation \
	--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 0.0005 --stop-min-lr '1e-09' \
	--lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
	--dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy \
	--label-smoothing 0.1 --max-tokens $MAXTOKENS --save-dir $SAVEDIR \
	--tensorboard-logdir $SAVEDIR --fp16 --update-freq $UPDATEFREQ --max-update 1200000 \
	--save-interval 1 --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
	--eval-bleu --eval-bleu-remove-bpe sentencepiece --max-tokens-valid 2000 \
	--eval-bleu-args '{"beam": 1, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok=space \
	--source-lang $SRC --target-lang $TGT --sacrebleu-tokenizer $TOKENIZER --keep-last-epochs 5 \
	--multi-waitk --eval-waitk $WAITK $EXTRA_ARGS --share-all-embeddings \
    --language-model-name $LANGUAGE_MODEL_NAME --encoder-lm-dropout --encoder-lm-dropout-ratio 0.5 
```

#### Run inference of word-level wait-k models:
Use the following command.  
```shell
CUDA_VISIBLE_DEVICES=0 python fairseq-generate $DATA_PATH \
    --path $MODEL_PATH --max-tokens 3000 --gen-subset test \
    --scoring sacrebleu --results-path $RESULT_PATH/${SAVE_FOLDER}_k$k \
    --source-lang $SRC --target-lang $TGT --sacrebleu-tokenizer intl \
    --task waitk_translation --eval-waitk $k --fp16 --beam 1 --word-waitk \
    --user-dir $PATH_TO_FAIRSEQ/examples/word_waitk
```

#### Run inference of word-level wait-k + LM models:
Use the following command.  
`LANGUAGE_MODEL_NAME` should be the same one as the one used in training.
```shell
CUDA_VISIBLE_DEVICES=0 python fairseq-generate $DATA_PATH \
    --path $MODEL_PATH --max-tokens 3000 --gen-subset test \
    --scoring sacrebleu --results-path $RESULT_PATH/${SAVE_FOLDER}_k$k \
    --source-lang $SRC --target-lang $TGT --sacrebleu-tokenizer intl \
    --task waitk_translation --eval-waitk $k --fp16 --beam 1 --word-waitk \
    --language-model-name $LANGUAGE_MODEL_NAME \
    --user-dir $PATH_TO_FAIRSEQ/examples/word_waitk
```

#### Measure BLEU and average lagging:
Use the following command.  
`PATH_TO_GENERATED_FILE` is the path to the `generate-test.txt`.  
`PATH_TO_REFERENCE_FILE` is the path to the raw reference file, e.g., `test.de-en.en`
```shell
python measure_delay.py \
    --generated-file $PATH_TO_GENERATED_FILE \
    -k $k -l $TGT \
    --reference-file $PATH_TO_REFERENCE_FILE
```
