#!/bin/bash

# download dependencies
./download_linux_dependencies.sh

# install dependencies
./install_linux_dependencies.sh

# make HC-Search
make

# after this, run avatol_crf_system_setup in MATLAB
