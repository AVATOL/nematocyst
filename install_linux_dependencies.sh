#!/bin/bash

#################################################################################################################
# This script downloads and installs all the dependencies required for HC-Search and the AVATOL system to work.
#################################################################################################################

### LINUX DEPENDENCIES REQUIRED FOR VOWPAL WABBIT BUILD ###
#sudo apt-get install libtool automake libboost-program-options1.49-dev build-essential libatlas-base-dev zlib1g-dev

### CREATE EXTERNAL DIRECTORY ###
mkdir -p external

### DOWNLOAD AND UNZIP/UNTAR ###

# LIBLINEAR
wget http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/liblinear.cgi?+http://www.csie.ntu.edu.tw/~cjlin/liblinear+zip -O external/liblinear.zip
unzip external/liblinear.zip -d external
mv external/liblinear-1.94 external/liblinear

# LIBSVM
wget http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+zip  -O external/libsvm.zip
unzip external/libsvm.zip -d external
mv external/libsvm-3.18 external/libsvm

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

#### AUTOMATIC INSTALL/MAKE ###

# LIBLINEAR
cd external/liblinear
make
cd -

# LIBSVM
cd external/libsvm
make
cd -

# VOWPAL WABBIT
cd external/vowpal_wabbit
./autogen.sh
make
cd -
