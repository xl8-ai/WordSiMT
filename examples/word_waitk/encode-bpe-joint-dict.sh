#!/bin/bash

DATAPATH=$1
SRC=$2
TGT=$3
BPESIZE=$4
DESTDIR=$5

ROOT=$(dirname "$0")
SCRIPTS=$ROOT/../../scripts
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py

#TRAIN_MINLEN=1  # remove sentences with <1 BPE token
#TRAIN_MAXLEN=250  # remove sentences with >250 BPE tokens

# learn BPE with sentencepiece
BPE_FILES="$DATAPATH/train.${SRC}-${TGT}.${SRC}","$DATAPATH/train.${SRC}-${TGT}.${TGT}"

echo "learning joint BPE over ${BPE_FILES}..."
python "$SPM_TRAIN" \
    --input=$BPE_FILES \
    --model_prefix=$DATAPATH/sentencepiece.bpe \
    --vocab_size=$BPESIZE \
    --character_coverage=1.0 \
    --model_type=bpe \
    --normalization_rule_name=identity \
    --input_sentence_size=8000000 \
    --shuffle_input_sentence \
    --treat_whitespace_as_suffix

echo "encoding train with learned BPE..."
python "$SPM_ENCODE" \
    --model "$DATAPATH/sentencepiece.bpe.model" \
    --output_format=piece \
    --inputs $DATAPATH/train.${SRC}-${TGT}.${SRC} $DATAPATH/train.${SRC}-${TGT}.${TGT} \
    --outputs $DATAPATH/train.bpe.joint.${BPESIZE}.${SRC}-${TGT}.${SRC} $DATAPATH/train.bpe.joint.${BPESIZE}.${SRC}-${TGT}.${TGT} \

echo "encoding valid with learned BPE..."
python "$SPM_ENCODE" \
    --model "$DATAPATH/sentencepiece.bpe.model" \
    --output_format=piece \
    --inputs $DATAPATH/val.${SRC}-${TGT}.${SRC} $DATAPATH/val.${SRC}-${TGT}.${TGT} \
    --outputs $DATAPATH/val.bpe.joint.${BPESIZE}.${SRC}-${TGT}.${SRC} $DATAPATH/val.bpe.joint.${BPESIZE}.${SRC}-${TGT}.${TGT} \

TEST_SOURCE="$DATAPATH/test.${SRC}-${TGT}.${SRC}"
TEST_TARGET="$DATAPATH/test.${SRC}-${TGT}.${TGT}"

if test -f "$TEST_SOURCE"; then
    echo "encoding test with learned BPE..."
    python "$SPM_ENCODE" \
	        --model "$DATAPATH/sentencepiece.bpe.model" \
	        --output_format=piece \
	        --inputs $TEST_SOURCE $TEST_TARGET \
	        --outputs $DATAPATH/test.bpe.joint.${BPESIZE}.${SRC}-${TGT}.${SRC} $DATAPATH/test.bpe.joint.${BPESIZE}.${SRC}-${TGT}.${TGT} \
fi

echo "preparing LM data..."
cp $DATAPATH/train.${SRC}-${TGT}.${SRC} $DATAPATH/train.bpe.joint.${BPESIZE}.${SRC}-${TGT}.lm.${SRC}
cp $DATAPATH/val.${SRC}-${TGT}.${SRC} $DATAPATH/val.bpe.joint.${BPESIZE}.${SRC}-${TGT}.lm.${SRC}
cp $DATAPATH/test.${SRC}-${TGT}.${SRC} $DATAPATH/test.bpe.joint.${BPESIZE}.${SRC}-${TGT}.lm.${SRC}
