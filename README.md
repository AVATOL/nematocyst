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

1. Create a folder to put your dataset in; call this folder `$DATASET$`. Create additional folders: `$DATASET$/Annotations`, `$DATASET$/Images` and `$DATASET$/Splits`
2. Put the images (.jpg) in the `Images` folder and the annotations (groundtruth masks, also .jpg) in the `Annotations` folder. Corresponding image and annotation files must have the same file name!
3. Create `Train.txt`, `Validation.txt` and `Test.txt` in the `Splits` folder. `Train.txt` is a list of images used for training the system. `Validation.txt` is a list of images used in the system to validate the trained model. `Test.txt` is a list of images for the system to score characters. For all these list of images, do not add the file extension to the name.

### Pre-Processing Step

Note: MATLAB is required.

1. Create a folder to put the preprocessed dataset in; call this folder `$PREPROCESSED$`.
2. Open MATLAB, make sure VLFeat is set up properly (see Installation Instructions), and run the following command in MATLAB. This should create files and folders in the `$PREPROCESSED$` folder.
```
preprocess('$DATASET$/Images', '$DATASET$/Annotations', '$DATASET$/Splits', '$PREPROCESSED$' )
```

### Training From Annotations

1. Create a folder to put the results from HC-Search; call this folder `$RESULTS$`.
2. Run the following command from the command line:
	`./HCSearch $PREPROCESSED$ $RESULTS$ 5 --learn H --learn C`
This should create some files in `$RESULTS$/models`

### Scoring Characters

Run the following command from the command line:
	`./HCSearch $PREPROCESSED$ $RESULTS$ 5 --infer HC`
This should create some files in `$RESULTS$/results`
	
### Post-Processing Tasks

Note: MATLAB is required.

1. Create a folder to put the postprocessed dataset in; call this folder `$POSTPROCESSED$`.
2. Open in MATLAB, make sure LIBSVM is set up properly, and run the following command in MATLAB. This should create visualization files and folders in the `$POSTPROCESSED$` folder.
```
visualize_results('$ROOT$/DataRaw/SomeDataset', '$ROOT$/DataPreprocessed/SomeDataset', '$ROOT$/Results/SomeExperiment', '$ROOT$/ResultsPostprocessed/SomeExperiment', 0:4, 0);
```

## Installation Instructions

Let `$ROOT$` denote the root directory containing `src` and this README.

### Windows

1. Before compiling the source or running the binary executable, dependencies must be installed in the `$ROOT$/external/` directory.
	- Eigen matrix libary
		1. Download from http://eigen.tuxfamily.org
			- Version 3.2.0 officially supported. Later versions should also work.
		2. Unpack to `$ROOT$/external/Eigen`
			- Unpack the directory structure such that this file path is valid: `$ROOT$/external/Eigen/Eigen/Dense`
	- SVM-Rank
		1. Download from http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html
		2. Unpack to `$ROOT$/external/svm_rank`
			- Unpack the directory structure such that the Makefile is in `$ROOT$/external/svm_rank`
	- LIBLINEAR
		1. Download from http://www.csie.ntu.edu.tw/~cjlin/liblinear/
			- Version 1.94 officially supported. Later versions should also work.
		2. Unpack to `$ROOT$/external/liblinear`
			- Unpack the directory structure such that the Makefile is in `$ROOT$/external/liblinear`
	- LIBSVM (Optional - if you use postprocessing modules OR if you use SVM for initial state in HCSearch)
		1. Download from http://www.csie.ntu.edu.tw/~cjlin/libsvm/
			- Version 3.17 officially supported. Later versions should also work.
		2. Unpack to `$ROOT$/external/libsvm`
			- Unpack the directory structure such that the Makefile is in `$ROOT$/external/libsvm`
		3. Set up in MATLAB (optional - if you use postprocessing modules):
			1. Open MATLAB, navigate in MATLAB to `$ROOT$/external/libsvm/matlab`
			2. Run the `make` command in MATLAB.
			3. Add `$ROOT$/external/libsvm/matlab` to your include paths in MATLAB.
	- VLFeat (Optional - if you use preprocessing modules)
		1. Download from http://www.vlfeat.org/
			- Version 0.9.18 officially supported. Later versions should also work.
		2. Unpack to `$ROOT$/external/vlfeat`
			- Unpack the directory structure such that this file path is valid: `$ROOT$/external/vlfeat/toolbox/vl_setup.m`
		3. Set up in MATLAB:
			1. Open MATLAB, add `$ROOT$/external/vlfeat/toolbox` to your include paths in MATLAB.
			2. Run the `vl_setup` command in MATLAB.
			3. Verify it is set up properly by running the `vl_version` command in MATLAB.
2. Run the provided binary executable `HCSearch.exe` (make sure it is in the top-most directory containing this README).
3. If you prefer to compile from source...
	1. Open `$ROOT$/src/HCSearch.sln` in Microsoft Visual Studio 2012 or later.
	2. Build the solution. Make sure it is on Release.
	3. Move `$ROOT$/src/Release/HCSearch.exe` to `$ROOT$/HCSearch.exe`.

### Linux

1. Before compiling the source or running the binary executable, dependencies must be installed in the `$ROOT$/external/` directory.
	- Eigen matrix libary
		1. Download from http://eigen.tuxfamily.org
			- Version 3.2.0 officially supported. Later versions should also work.
		2. Unpack to `$ROOT$/external/Eigen`
			- Unpack the directory structure such that this file path is valid: `$ROOT$/external/Eigen/Eigen/Dense`
	- SVM-Rank
		1. Download from http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html. Make sure to download the source code version.
		2. Unpack to `$ROOT$/external/svm_rank`
			- Unpack the directory structure such that the Makefile is in `$ROOT$/external/svm_rank`
		3. Compile by running `make` in `$ROOT$/external/svm_rank`
	- LIBLINEAR
		1. Download from http://www.csie.ntu.edu.tw/~cjlin/liblinear/
			- Version 1.94 officially supported. Later versions should also work.
		2. Unpack to `$ROOT$/external/liblinear`
			- Unpack the directory structure such that the Makefile is in `$ROOT$/external/liblinear`
		3. Compile by running `make` in `$ROOT$/external/liblinear`
	- LIBSVM (Optional - if you use postprocessing modules OR if you use SVM for initial state in HCSearch)
		1. Download from http://www.csie.ntu.edu.tw/~cjlin/libsvm/
			- Version 3.17 officially supported. Later versions should also work.
		2. Unpack to `$ROOT$/external/libsvm`
			- Unpack the directory structure such that the Makefile is in `$ROOT$/external/libsvm`
		3. Compile by running `make` in `$ROOT$/external/libsvm`
		4. Set up in MATLAB (optional - if you use postprocessing modules):
			1. Open MATLAB, navigate in MATLAB to `$ROOT$/external/libsvm/matlab`
			2. Run the `make` command in MATLAB.
			3. Add `$ROOT$/external/libsvm/matlab` to your include paths in MATLAB.
	- VLFeat (Optional - if you use preprocessing modules)
		1. Download from http://www.vlfeat.org/
			- Version 0.9.18 officially supported. Later versions should also work.
		2. Unpack to `$ROOT$/external/vlfeat`
			- Unpack the directory structure such that this file path is valid: `$ROOT$/external/vlfeat/toolbox/vl_setup.m`
		3. Set up in MATLAB:
			1. Open MATLAB, add `$ROOT$/external/vlfeat/toolbox` to your include paths in MATLAB.
			2. Run the `vl_setup` command in MATLAB.
			3. Verify it is set up properly by running the `vl_version` command in MATLAB.
2. Compile from source by running `make` in the `$ROOT$` directory. It should create the binary file `$ROOT$/HCSearch`.

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
				CRH: learn cost with random H
				ALL: short-hand for H, C, COH
				(none): short-hand for H, C
	--infer arg	: inference
				HC: learned heuristic and cost
				HL: learned heuristic and oracle cost
				LC: oracle heuristic and learned cost
				LL: oracle heuristic and cost
				RL: random heuristic and oracle cost
				RC: random heuristic and learned cost
				ALL: short-hand for HC, HL, LC, LL
				(none): short-hand for HC

Advanced options:
	--anytime arg		: turn on saving anytime predictions if true
	--beam-size arg		: beam size for beam search
	--bound-successor arg	: maximum number of successor candidates (default=1000)
	--cut-mode arg		: edges|state (cut edges by edges independently or by state)
	--cut-param arg		: temperature parameter for stochastic cuts
	--num-test-iters arg	: number of test iterations
	--num-train-iters arg	: number of training iterations
	--learner arg		: svmrank|online
	--save-features arg	: save rank features during learning if true
	--save-mask arg		: save final prediction label masks if true
	--search arg		: greedy|breadthbeam|bestbeam
	--splits-path arg	: specify alternate path to splits folder
	--successor arg		: flipbit|flipbit-neighbors|stochastic|stochastic-neighbors|stochastic-confidences-neighbors|cut-schedule|cut-schedule-neighbors|cut-schedule-confidences-neighbors
	--unique-iter arg	: unique iteration ID (num-test-iters needs to be 1)
	--verbose arg		: turn on verbose output if true

Notes:
* The first three arguments are required. They are the input directory, output directory and time bound.
* Can use multiple --infer and --learn options in any order to define a schedule. Must come after the mandatory arguments.
```

### Preprocessing Module

The main preprocessing module is `preprocess/preprocess.m`:

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
%                   e.g. 'DataRaw/SomeDataset/Annotations'
%	splitsPath:	folder path that contains Train.txt,
%               Validation.txt, Test.txt
%                   e.g. 'DataRaw/SomeDataset/Splits'
%	outputPath:	folder path to output preprocessed data
%                   e.g. 'DataPreprocessed/SomeDataset'
%
%	allData:	data structure containing all preprocessed data
```

### Postprocessing Module

The main visualization postprocessing module is `postprocess/visualize_results.m`:

```
function visualize_results( rawDir, preprocessedDir, resultsDir, outputDir, label2color, timeRange, foldRange, searchTypes, splitsName, allData )
%VISUALIZE_RESULTS Visualize results folder.
%
%	rawDir:             folder path containing original images and annotations
%                           e.g. 'DataRaw/SomeDataset'
%	preprocessedDir:	folder path containing preprocessed data
%                           e.g. 'DataPreprocessed/SomeDataset'
%	resultsDir:         folder path containing HC-Search results
%                           e.g. 'Results/SomeExperiment'
%	outputDir:          folder path to output visualization
%                           e.g. 'ResultsPostprocessed/SomeExperiment'
%   label2color:        mapping from labels to colors (use containers.Map)
%   timeRange:          range of time bound
%   foldRange:          range of folds
%   searchTypes:        list of search types 1 = HC, 2 = HL, 3 = LC, 4 = LL
%   splitsName:         (optional) alternate name to splits folder
%	allData:            (optional) data structure containing all preprocessed data
```

## Technical Documentation

Technical documentation (i.e. how to use the C++ API) is available in the `doc` directory.

If not available, then use Doxygen (doxygen.org) to generate the documentation from source code. Run the command `doxygen` from the command line in the directory containing the `Doxyfile`.