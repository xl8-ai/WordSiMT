#!/bin/bash

DATAPATH=$1
SRC=$2
TGT=$3
BPESIZE=$4
DESTDIR=$5
BERT_MODEL_NAME=$6

./encode-bpe-joint-dict.sh "$@"
./preprocess-joint-dict.sh "$@"
