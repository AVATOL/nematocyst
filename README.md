HC-SEARCH::CV
==========

HC-Search Structured Prediction Framework for Computer Vision

## Introduction

HC-Search is a learning and inference framework for structured prediction. This is an implementation of HC-Search primarily geared toward scene labeling for computer vision, but may be adapted to other computer vision tasks.

The HC-Search framework allows you to define a search space and search procedure to perform structured prediction. HC-Search learns a heuristic function to uncover high quality candidates and a cost function to select the best candidate.

A search space consists of a heuristic feature function, cost feature function, initial state function, successor function and loss function. A search procedure can be anything that you define, including the commonly used greedy search and beam search. Our implementation allows you to define all of these components.

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

## Overview

Usually the pipeline is along the lines of this:

Input: Images, groundtruth, and training/validation/test splits

1. (Preprocessing Modules) Extract features from images and put into format ready for the HCSearch program.
2. (HC-Search Learning/Inference) Run HCSearch to learn model and perform inference. Sample usage: ./HCSearch INPUT_DIR OUTPUT_DIR TIMEBOUND --learn --infer
3. (Postprocessing Modules) Convert result files from HC-Search into format suitable for visualization, evaluation and further analysis.

Output: Evaluation, visualization and analysis of inference on test images

## Quick Demo

If you are just interested in getting something running, then this is the section. Please first read the Installation Instructions section.

### Setup Images, Groundtruth and Train/Validation/Test Splits

1. Create a folder $ROOT/DataRaw/SomeDataset/. Create additional folders in it: Groundtruth/, Images/ and Splits/
2. Put the images (.jpg) in the Images/ folder and the groundtruth masks (.jp) in the Groundtruth folder. Corresponding images/groundtruth must have the same file name!
3. Create Train.txt, Validation.txt and Test.txt in Splits/. In each file, list the file name in the Images folder (without .jpg extension) that belong in each split you want.

### Preprocessing

1. Create folder $ROOT/DataPreprocessed/.
2. Open MATLAB, make sure VLFeat is set up properly (run vl_setup), and run the following command in Matlab: 
preprocess('$ROOT/DataRaw/SomeDataset/Images', '$ROOT/DataRaw/SomeDataset/Groundtruth', '$ROOT/DataRaw/SomeDataset/Splits', '$ROOT/DataPreprocessed/SomeDataset' )
This should create files and folders in the $ROOT/DataPreprocessed/SomeDataset/ folder.

### HC-Search Learn/Infer

1. Run the following command from the command line: ./HCSearch $ROOT/DataPreprocessed/SomeDataset $ROOT/Results 5 --learn --infer
This learns a heuristic and cost function and then runs LL/HL/LC/HL search. 
This should create files and folders in $ROOT/Results/

### Postprocessing

Coming soon.

## Preprocessing Module

The preprocessing modules are written in MATLAB. Before running them, make sure to set up VLFeat:

1. Download and extract VLFeat to $ROOT/external/vlfeat/ (or some other location) if not already done.
2. Launch MATLAB. Run the following command in MATLAB: run('$ROOT/external/vlfeat/toolbox/vl_setsup') or wherever it is installed.
	- Alternatively, add the line to your startup.m file to automatically run this everytime MATLAB starts.
3. Verify is set up properly by running the following command in MATLAB: vl_version

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

## HC-Search Command Line Options

```
Program usage: ./HCSearch INPUT_DIR OUTPUT_DIR TIMEBOUND [--learn [H|C|COH]]* [--infer [HC|HL|LC|LL]]* ... [--option=value]
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
	--anytime arg		: turn off saving anytime predictions if false
	--beamsize arg		: beam size for beam search
	--cutparam arg		: temperature parameter for stochastic cuts
	--learner arg		: svmrank|online
	--savefeatures arg	: save rank features during learning if true
	--search arg		: greedy|breadthbeam|bestbeam
	--splitspath arg	: specify alternate path to splits folder
	--successor arg		: flipbit|stochastic

Notes:
* The first three arguments are required. They are the input directory, output directory and time bound.
* Can use multiple --infer and --learn options in any order to define a schedule. Must come after the mandatory arguments.
```

## Postprocessing Module

Coming soon.

## Technical Documentation

Technical documentation (i.e. how to use the C++ API) is available in the $ROOT/doc/ directory.

If not available, then use Doxygen (doxygen.org) to generate the documentation from source code. Use the provided Doxyfile. Run the command doxygen from the command line in the $ROOT/ directory.