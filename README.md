nematocyst
=============

High-Clutter Character Segmentation on Nematocysts

## Introduction

This software was developed for learning to recognize basal tubule, distal tubule and capsule characters of nematocysts. However, it is intended to work for any (in theory) images of biological specimens that can be cast into a scene labeling framework with simple enough characters for scoring. Currently the software only supports segmenting out the characters from images and not scoring the characters.

The user provides images and annotations of nematocysts for the system to learn how to detect/segment characters. Once the system is done training, the user can provide new images for the software to detect/segment the characters.

The system follows this pipeline: 1) feature extraction, 2) training for detection/scoring, 3) character detection/segmentation. Character detection is performed through a framework called HC-Search. HC-Search is a structured prediction framework for computer vision. It was found more effective over traditional conditional random field (CRF) approaches for cluttered biological images.

## How to Use Software

This system was built to interface with the avatol\_cv software. Please consult the documentation for the avatol\_cv project.

While it is meant to run as a module through the avatol\_cv software, it could also be run as a stand-alone module. However, this is not recommended. Here are some rough steps if you are interested:

1. Preprocess: `preprocess_avatol()` (MATLAB)
2. Segmentation: `./HCSearch`
3. Postprocess: `postprocess_avatol()` (MATLAB)

## Installation Instructions

Currently supports Windows and Mac. Linux is unsupported but installation should still work on Linux.

### One-Step Install (Recommended)

You can automate the entire installation process by running the following (Python 2.7 is required):

```
python setup.py
```

This should work on Windows and Mac at least.

### Manual Installation

If you prefer to manually install, here are some rough instructions. The one-step script above essentially does the following:

1. Download dependencies: Eigen, LIBLINEAR, LIBSVM, VLFeat, Vowpal Wabbit
2. Build the dependencies if using Mac/Linux, otherwise use the Windows binaries
3. Build HC-Search if using Mac/Linux, otherwise use the provided Windows binary

#### 1) Download Dependencies

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
	1. Download or git clone from https://github.com/JohnLangford/vowpal\_wabbit
		- Version 7.7 (git tag b2b702c) officially supported. Later versions should also work.
	2. Unpack to `$ROOT$/external/vowpal_wabbit`
		- Unpack the directory structure such that this file path is valid: `$ROOT$/external/vowpal_wabbit/Makefile`

#### 2) Build the dependencies

Build the dependencies: LIBLINEAR, LIBSVM, Vowpal Wabbit. For LIBLINEAR and LIBSVM, follow their instructions. Build the command-line tools _and_ MATLAB interfaces.

For Vowpal Wabbit, follow their instructions as well. If you have problems building Vowpal Wabbit, you may need to install additional Linux dependencies beforehand:

```
sudo apt-get install libtool
sudo apt-get install automake
sudo apt-get install libboost-program-options1.49-dev
sudo apt-get install build-essential
sudo apt-get install libatlas-base-dev
sudo apt-get install zlib1g-dev
```

The dependencies are analogous for Mac. Use Homebrew. Something like this:

```
brew install autoconf
brew install automake
brew install libtool
brew install boost
```

(Verify that glibtoolize exists before configuring/compiling VowpalWabbit on Mac.)

If you are using Windows, you do not need to build anything _except_ for the LIBLINEAR and LIBSVM MATLAB interfaces. Just make sure the following command-line binaries are working:
- `$ROOT$/external/vw.exe`
- `$ROOT$/external/liblinear/windows/predict.exe`
- `$ROOT$/external/liblinear/windows/train.exe`

#### 3) Build HC-Search

If you are using Mac/Linux, simply run `make` to build the HC-Search module.

If you are using Windows, you do not need to build it. Just make sure the included binary `$ROOT$/HCSearch.exe` is working.

If you do want to build it for Windows, you will need Visual Studio 2012. Open `$ROOT$/src/HCSearch.sln` and build the solution.

Note: The included binary `$ROOT$/HCSearch.exe` was built _without_ MPI functionality. The included binary `$ROOT$/HCSearchMPI.exe` was built _with_ MPI functionality. 

#### Using MPI

Using MPI can improve performance by running some tasks in parallel, but this feature is optional and is only intended for advanced users. It is also mainly intended for computing clusters rather than personal computers.

For Windows, if you plan to use MPI, you do not need to build the HC-Search module. Just make sure the included binary `$ROOT$/HCSearchMPI.exe` is working.

If the executable complains about a missing DLL related to MPI or you would like to build the HC-Search module from scratch using MPI, you will need to install MPI on your machine. You can download the installer here: http://www.microsoft.com/en-us/download/details.aspx?id=41634

Note: The default build settings will try to compile with MPI (Message Passing Interface for parallel processing) using Microsoft's HPC Pack. If you do not want to build with MPI, open up the properties of each project in the solution and remove the `USE_MPI` preprocessor flag.

For Mac/Linux, you will need to install MPI to build from source. You can run the following command for Linux:

```
sudo apt-get install libcr-dev mpich2 mpich2-doc
```

The dependencies are analogous for Mac. Use Homebrew.

Then run `make mpi` instead of `make`.

