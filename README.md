nematocyst
=============

Character Recognition on Nematocysts

## Introduction

This software is used for learning to recognize basal and distal tubule characters of nematocysts. The user provides images and annotations of nematocysts for the system to learn initially. Once the system is done training, the user can provide new images for the software to recognize the characters.

The system consists of three modules: pre-processing, HC-Search and post-processing. The pre-processing module extracts computer vision features from the input images and puts this in a suitable input format for the HC-Search module. The HC-Search module is responsible for learning and inferring characters from images. In learning mode, HC-Search learns a model of the nematocyst from the input training images and annotations. In inference mode, HC-Search uses the learned model to recognize characters from input test images. Finally, the post-processing module provides utilities to visualize and evaluate the scored characters.

HC-Search is a structured prediction framework for computer vision. It was found more effective over traditional conditional random field (CRF) approaches.

## How to Use the Software

### Installation

Please see the Installation Instructions section before proceeding.

### Images and Annotation Setup

1. Create a folder to put your dataset in; call this folder $DATASET. Create additional folders in it: $DATASET/Annotations/, $DATASET/Images/ and $DATASET/Splits/
2. Put the images (.jpg) in the Images folder and the annotations (groundtruth masks, also .jpg) in the Annotations folder. Corresponding image and annotation files must have the same file name!
3. Create Train.txt, Validation.txt and Test.txt in the Splits folder. Train.txt is a list of images used for training the system. Validation.txt is a list of images used in the system to validate the trained model. Test.txt is a list of images for the system to score characters. For all these list of images, do not add the file extension to the name.

### Pre-Processing Step

Note: MATLAB is required.

1. Create a folder to put the preprocessed dataset in; call this folder $PREPROCESSED.
2. Open MATLAB, make sure VLFeat is set up properly (see Installation Instructions), and run the following command in MATLAB: 
	preprocess('$DATASET/Images', '$DATASET/Annotations', '$DATASET/Splits', '$PREPROCESSED' )
This should create files and folders in the $PREPROCESSED folder.

### Training From Annotations

1. Create a folder to put the results from HC-Search; call this folder $RESULTS.
2. Run the following command from the command line:
	./HCSearch $PREPROCESSED $RESULTS 5 --learn H --learn C
This should create some files in $RESULTS/models

### Scoring Characters

Run the following command from the command line:
	./HCSearch $PREPROCESSED $RESULTS 5 --infer HC
This should create some files in $RESULTS/results
	
### Post-Processing Tasks

Note: MATLAB is required.

In development.

## Installation Instructions

Let $ROOT denote the root directory containing src/.

### Windows

1. Before compiling the source or running the binary executable, dependencies must be installed in the $ROOT/external/ directory.
	- Eigen matrix libary
		1. Download from http://eigen.tuxfamily.org
		2. Unpack to $ROOT/external/Eigen/
	- SVM-Rank
		1. http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html
		2. Unpack to $ROOT/external/svm_rank/
	- LIBLINEAR
		1. Download from http://www.csie.ntu.edu.tw/~cjlin/liblinear/
		2. Unpack to $ROOT/external/liblinear/
	- LIBSVM (Optional - if you use SVM for initial state in HCSearch)
		1. Download from http://www.csie.ntu.edu.tw/~cjlin/libsvm/
		2. Unpack to $ROOT/external/libsvm/
	- VLFeat (Optional - if you use preprocessing modules)
		1. Download from http://www.vlfeat.org/
		2. Unpack to $ROOT/external/vlfeat/
2. Run the provided binary executable HCSearch.exe (make sure it is in $ROOT/external/).
3. If you prefer to compile from source...
	1. Open $ROOT/src/HCSearch.sln in Microsoft Visual Studio 2012 or later.
	2. Build the solution. Make sure it is on Release.
	3. Move $ROOT/src/Release/HCSearch.exe to $ROOT/HCSearch.exe.

### Linux

1. Before compiling the source or running the binary executable, dependencies must be installed in the external/ directory.
	- Eigen matrix libary
		1. Download from http://eigen.tuxfamily.org
		2. Unpack to $ROOT/external/Eigen/
	- SVM-Rank
		1. Download from http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html. Make sure to download the source code version.
		2. Unpack to $ROOT/external/svm_rank/
		3. Compile by running make in $ROOT/external/svm_rank/
	- LIBLINEAR
		1. Download from http://www.csie.ntu.edu.tw/~cjlin/liblinear/
		2. Unpack to $ROOT/external/liblinear/
		3. Compile by running make in $ROOT/external/liblinear/
	- LIBSVM (Optional - if you use SVM for initial state in HCSearch)
		1. Download from http://www.csie.ntu.edu.tw/~cjlin/libsvm/
		2. Unpack to $ROOT/external/libsvm/
		3. Compile by running make in $ROOT/external/libsvm/
	- VLFeat (Optional - if you use preprocessing modules)
		1. Download from http://www.vlfeat.org/
		2. Unpack to $ROOT/external/vlfeat/
2. Compile from source by running make in $ROOT/.

## Reference

### HC-Search Command Line Options

```
Program usage: ./HCSearch INPUT_DIR OUTPUT_DIR TIMEBOUND [--learn (H|C|COH)]* [--infer (HC|HL|LC|LL)]* ... [--option=value]
Main options:
	--help		: produce help message
	--demo		: run the demo program (ignores --learn and --infer)
	--learn arg	: learning
				H: learn heuristic
				C: learn cost
				COH: learn cost with oracle H
				(none): short-hand for H, C, COH
	--infer arg	: inference
				HC: learned heuristic and cost
				HL: learned heuristic and oracle cost
				LC: oracle heuristic and learned cost
				LL: oracle heuristic and cost
				(none): short-hand for HC, HL, LC, LL

Advanced options:
	--anytime arg			: turn on  saving anytime predictions if true
	--beam-size arg			: beam size for beam search
	--cut-param arg			: temperature parameter for stochastic cuts
	--num-test-iters arg	: number of test iterations
	--num-train-iters arg	: number of training iterations
	--learner arg			: svmrank|online
	--save-features arg		: save rank features during learning if true
	--search arg			: greedy|breadthbeam|bestbeam
	--splits-path arg		: specify alternate path to splits folder
	--successor arg			: flipbit|stochastic

Notes:
* The first three arguments are required. They are the input directory, output directory and time bound.
* Can use multiple --infer and --learn options in any order to define a schedule. Must come after the mandatory arguments.
```

### Preprocessing Module

The main preprocessing module is $ROOT/preprocess/preprocess.m:

```
function [ allData ] = preprocess( imagesPath, labelsPath, splitsPath, outputPath )
%PREPROCESS Preprocesses folders of images and groundtruth labels into a
%data format for HCSearch to work. Performs feature extraction.
%
%This implementation creates a regular grid of HOG/SIFT patches.
%Therefore only need grayscale images.
%
%One can change the features (e.g. add color) by editing this file.
%
%	imagesPath:	folder path to images folder of *.jpg images
%                   e.g. 'DataRaw/SomeDataset/Images'
%	labelsPath:	folder path to groundtruth folder of *.jpg label masks
%                   e.g. 'DataRaw/SomeDataset/Groundtruth'
%	splitsPath:	folder path that contains Train.txt,
%               Validation.txt, Test.txt
%                   e.g. 'DataRaw/SomeDataset/Splits'
%	outputPath:	folder path to output preprocessed data
%                   e.g. 'DataPreprocessed/SomeDataset'
%
%	allData:	data structure containing all preprocessed data
```

## Technical Documentation

Technical documentation (i.e. how to use the C++ API) is available in the doc/ directory.

If not available, then use Doxygen (doxygen.org) to generate the documentation from source code. Use the provided Doxyfile. Run the command doxygen from the command line in the directory containing the Doxyfile.