hcsearchcv
==========

HC-Search Learning and Inference Framework for Computer Vision

## Introduction

HC-Search is a learning and inference framework for structured prediction. This is an implementation of HC-Search primarily geared toward scene labeling for computer vision, but may be adapted to other computer vision tasks.

The HC-Search framework allows you to define a search space and search procedure to perform structured prediction. HC-Search learns a heuristic function to uncover high quality candidates and a cost function to select the best candidate.

A search space consists of heuristic feature function, cost feature function, initial state function, successor function and loss function. A search procedure can be anything that you define, including the commonly used greedy search and beam search. Our implementation allows you to define all of these components.

## Installation Instructions

Let $ROOT denote the root directory containing src/.

### Windows

1. Before compiling the source or running the binary executable, dependencies must be installed in the external/ directory.
	- Eigen matrix libary
		1. Download from http://eigen.tuxfamily.org
		2. Unpack to $ROOT/external/Eigen/
	- SVM-Rank
		1. http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html
		2. Unpack to $ROOT/external/svm_rank/
	- LIBLINEAR
		1. Download from http://www.csie.ntu.edu.tw/~cjlin/liblinear/
		2. Unpack to $ROOT/external/liblinear/
	- LIBSVM
		1. Download from http://www.csie.ntu.edu.tw/~cjlin/libsvm/
		2. Unpack to $ROOT/external/libsvm/
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
	- LIBSVM
		1. Download from http://www.csie.ntu.edu.tw/~cjlin/libsvm/
		2. Unpack to $ROOT/external/libsvm/
		3. Compile by running make in $ROOT/external/libsvm/
2. Compile from source by running make in $ROOT.

## Command Line Options

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

## Technical Documentation

Technical documentation (i.e. how to use the C++ API) is available in the doc/ directory.

If not available, then use Doxygen (doxygen.org) to generate the documentation from source code. Use the provided Doxyfile.