nematocyst
=============

Character Recognition on Nematocysts

## Introduction

This software was developed for learning to recognize basal tubule, distal tubule and capsule characters of nematocysts. However, it is intended to work for any (in theory) images of biological specimens that can be cast into a scene labeling framework with simple enough characters for scoring.

The user provides images and annotations of nematocysts for the system to learn how to detect and score characters. Once the system is done training, the user can provide new images for the software to detect and score the characters.

The system follows this pipeline: 1) feature extraction, 2) training for detection/scoring, 3) character detection, 4) character scoring. Character detection is performed through a framework called HC-Search. HC-Search is a structured prediction framework for computer vision. It was found more effective over traditional conditional random field (CRF) approaches for cluttered biological images.

## How to Use the Software

### Installation

Please see the Installation Instructions section before proceeding.

### Input

The following should be the directory structure of the data set for input into the system:

- `SOME_DATASET_FOLDER/`
	- `media/`
		- `<mediaID>_<stuff>.jpg`
	- `annotations/`
		- `<mediaID>_<charID>.txt`
	- `sorted_input_data_<charID>_<charName>.txt`

The directory `annotations` contains annotations of this format:

```
x1,y1;...;xn,yn:charID:charName:charStateID:charStateName
...
```

The `sorted_input_data_<charID>_<charName>.txt` file specifies what is used for training or scoring. It follows this format:

```
training_data:media/<name_of_mediafile>:char_state:<pathname_of_annotation_file>:taxonID:<line_number>
...
image_to_score:media/<name_of_mediafile>:taxonID
...
```

### Output

The system will output the folowing files:

- `SOME_DATASET_FOLDER/`
	- `detection_results`
		- `<mediaID>_<charID>.txt`
	- `sorted_output_data_<charID>_<charName>.txt`

The directory `detection_results` contains the detected characters, in the same format as the annotation files.

The `sorted_output_data_<charID>_<charName>.txt` file contains the character scores and any images not scored. It follows this format:

```
training_data:media/<name_of_mediafile>:char_state:annotation/<name_of_annotation_file>
...
image_scored:media/<name_of_mediafile>:char_state:detection_results/<name_of_annotation_file>
...
image_not_scored:media/<name_of_mediafile>
...
```

### Running

Simply run the following command with the appropriate paths and names:

```
invoke_crf_system('path/to/sorted_input_data_<charID>_<charName>.txt', 'path/to/sorted_output_data_<charID>_<charName>.txt');
```

This command executes the pipeline: 1) feature extraction, 2) training for detection/scoring, 3) character detection, 4) character scoring.

For setting advanced parameters, pass a third parameter `options` to `invoke_crf_system()`.

## Installation Instructions

High level installation instructions:

1. Download dependencies: Eigen, LIBLINEAR, LIBSVM, VLFeat, Vowpal Wabbit
2. Build the dependencies if using Linux, otherwise use the Windows binaries
3. Build HC-Search if using Linux, otherwise use the provided Windows binary
4. Set up include paths in MATLAB using matlab_crf_install.m

Linux Bonus: If you are running Linux, you can automate steps 1, 2 and 3 by running the following:

```
./master_install_linux.sh
```

### 1) Download Dependencies

If you are using Linux, simply run the following script to download everything automatically:

```
./download_linux_dependencies.sh
```

Otherwise if you are using Windows or would like to manually download the files, following these instructions:

Let `$ROOT$` denote the root directory containing `src` and this README. Dependencies must be installed in the `$ROOT$/external/` directory.

- Eigen matrix libary
	1. Download from http://eigen.tuxfamily.org
		- Version 3.2.1 officially supported. Later versions should also work.
	2. Unpack to `$ROOT$/external/Eigen`
		- Unpack the directory structure such that this file path is valid: `$ROOT$/external/Eigen/Eigen/Dense`
- LIBLINEAR
	1. Download from http://www.csie.ntu.edu.tw/~cjlin/liblinear/
		- Version 1.94 officially supported. Later versions should also work.
	2. Unpack to `$ROOT$/external/liblinear`
		- Unpack the directory structure such that the Makefile is in `$ROOT$/external/liblinear`
- LIBSVM
	1. Download from http://www.csie.ntu.edu.tw/~cjlin/libsvm/
		- Version 3.18 officially supported. Later versions should also work.
	2. Unpack to `$ROOT$/external/libsvm`
		- Unpack the directory structure such that the Makefile is in `$ROOT$/external/libsvm`
- VLFeat
	1. Download from http://www.vlfeat.org/
		- Version 0.9.18 officially supported. Later versions should also work.
	2. Unpack to `$ROOT$/external/vlfeat`
		- Unpack the directory structure such that this file path is valid: `$ROOT$/external/vlfeat/toolbox/vl_setup.m`
- Vowpal Wabbit
	1. Download or git clone from https://github.com/JohnLangford/vowpal_wabbit
		- Version 7.7 (git tag b2b702c) officially supported. Later versions should also work.
	2. Unpack to `$ROOT$/external/vowpal_wabbit`
		- Unpack the directory structure such that this file path is valid: `$ROOT$/external/vowpal_wabbit/Makefile`

### 2) Build the dependencies

If you are using Linux, simply run the following script to build everything automatically:

```
./install_linux_dependencies.sh
```

If you have problems building Vowpal Wabbit, you may need to install additional Linux dependencies beforehand:

```
sudo apt-get install libtool automake libboost-program-options1.49-dev build-essential libatlas-base-dev zlib1g-dev
```

If you are using Windows, you do not need to build anything. Just make sure the following binaries are working:
- `$ROOT$/external/vowpal_wabbit/vw.exe`
- `$ROOT$/external/liblinear/windows/predict.exe`
- `$ROOT$/external/liblinear/windows/train.exe`

### 3) Build HC-Search

If you are using Linux, simply run `make` to build the HC-Search module.

If you are using Windows, you do not need to build it. Just make sure the included `$ROOT$/HCSearch.exe` binary is working.

If you do want to build it for Windows, you will need Visual Studio 2012. Open `$ROOT$/src/HCSearch.sln` and build the solution.

### 4) MATLAB Include Paths

Finally, open MATLAB and run `matlab_crf_install`. This will set up the include paths and install additional necessary MATLAB functionality.