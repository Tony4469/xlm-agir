# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

lg=$1  # input language

# data path
MAIN_PATH=$PWD
TOOLS_PATH=$PWD/tools

# tools
MOSES_DIR=$TOOLS_PATH/mosesdecoder
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fast
WMT16_SCRIPTS=$TOOLS_PATH/wmt16-scripts

# tools path
mkdir -p $TOOLS_PATH

#
# Download and install tools
#

cd $TOOLS_PATH

# Download Moses
if [ ! -d "$MOSES_DIR" ]; then
  echo "Cloning Moses from GitHub repository..."
  git clone https://github.com/moses-smt/mosesdecoder.git
fi

# Download fastBPE
if [ ! -d "$FASTBPE_DIR" ]; then
  echo "Cloning fastBPE from GitHub repository..."
  git clone https://github.com/glample/fastBPE
fi

# Compile fastBPE
if [ ! -f "$FASTBPE" ]; then
  echo "Compiling fastBPE..."
  cd fastBPE
  g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
  cd ..
fi

# Download Sennrich's tools
if [ ! -d "$WMT16_SCRIPTS" ]; then
  echo "Cloning WMT16 preprocessing scripts..."
  git clone https://github.com/rsennrich/wmt16-scripts.git
fi

# Download WikiExtractor
if [ ! -d $TOOLS_PATH/wikiextractor ]; then
    echo "Cloning WikiExtractor from GitHub repository..."
    git clone https://github.com/attardi/wikiextractor.git
fi

# # Download Model

# cd ..
# echo "Cloning Model from GitHub repository..."
# wget -c https://dl.fbaipublicfiles.com/XLM/mlm_tlm_xnli15_1024.pth
# ls
# pwd
# echo "yes !"



