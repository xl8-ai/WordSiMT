#!/bin/bash

DATAPATH=$1
SRC=$2
TGT=$3
BPESIZE=$4
DESTDIR=$5

./encode-bpe-joint-dict.sh "$@"
./preprocess-joint-dict.sh "$@"
