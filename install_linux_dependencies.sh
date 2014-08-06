#!/bin/bash

#################################################################################################################
# This script all the dependencies required for HC-Search and the AVATOL system to work.
#################################################################################################################

### LINUX DEPENDENCIES REQUIRED FOR VOWPAL WABBIT BUILD ###
#sudo apt-get install libtool automake libboost-program-options1.49-dev build-essential libatlas-base-dev zlib1g-dev

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
