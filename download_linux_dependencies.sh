#!/bin/bash

#################################################################################################################
# This script downloads all the dependencies required for HC-Search and the AVATOL system to work.
#################################################################################################################

### CREATE EXTERNAL DIRECTORY ###
mkdir -p external

### DOWNLOAD AND UNZIP/UNTAR ###

# LIBLINEAR
wget https://github.com/cjlin1/liblinear/archive/v210.zip -O external/liblinear.zip
unzip external/liblinear.zip -d external
mv external/liblinear-210 external/liblinear

# LIBSVM
wget https://github.com/cjlin1/libsvm/archive/v320.zip -O external/libsvm.zip
unzip external/libsvm.zip -d external
mv external/libsvm-320 external/libsvm

# EIGEN
wget http://bitbucket.org/eigen/eigen/get/3.2.1.tar.gz -O external/eigen.tar.gz
tar -xzvf external/eigen.tar.gz -C external
mv external/eigen-eigen-6b38706d90a9 external/Eigen

# VLFEAT
wget http://www.vlfeat.org/download/vlfeat-0.9.18-bin.tar.gz -O external/vlfeat.tar.gz
tar -xzvf external/vlfeat.tar.gz -C external
mv external/vlfeat-0.9.18 external/vlfeat

# VOWPAL WABBIT
wget https://github.com/JohnLangford/vowpal_wabbit/archive/7.7.tar.gz -O external/vowpal_wabbit.tar.gz
tar -xzvf external/vowpal_wabbit.tar.gz -C external
mv external/vowpal_wabbit-7.7 external/vowpal_wabbit
