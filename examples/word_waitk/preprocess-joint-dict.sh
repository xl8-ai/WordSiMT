#!/bin/bash

DATAPATH=$1
SRC=$2
TGT=$3
BPESIZE=$4
DESTDIR=$5
BERT_MODEL_NAME=$6

if [[ ! -z $BERT_MODEL_NAME ]]; then
	ADDITIONAL_ARGS="--language-model-name ${BERT_MODEL_NAME}"
fi

fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT} \
               	   --trainpref $DATAPATH/train.bpe.joint.${BPESIZE}.${SRC}-${TGT} \
	               --validpref $DATAPATH/val.bpe.joint.${BPESIZE}.${SRC}-${TGT} \
				   --testpref $DATAPATH/test.bpe.joint.${BPESIZE}.${SRC}-${TGT} \
		           --destdir ${DESTDIR}/${SRC}.${TGT}.${BPESIZE}.joint \
		           --joined-dictionary \
		           --workers 32 \
    			   --user-dir $PWD \
				   --task waitk_translation \
				   $ADDITIONAL_ARGS
