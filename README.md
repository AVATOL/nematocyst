hcsearchcv
==========

HC-Search Learning and Inference Framework for Computer Vision

## Introduction

HC-Search is a learning and inference framework for structured prediction. This is an implementation of HC-Search primarily geared toward scene labeling for computer vision, but may be adapted to other computer vision tasks.

The HC-Search framework allows you to define a search space and search procedure to perform structured prediction. HC-Search learns a heuristic function to uncover high quality candidates and a cost function to select the best candidate.

A search space consists of heuristic feature function, cost feature function, initial state function, successor function and loss function. A search procedure can be anything that you define, including the commonly used greedy search and beam search. Our implementation allows you to define all of these components.

## Compiling From Source

The source code is located in the src/ directory.

- If using Windows, use Visual Studio 2012 to build the solution.
- If using Linux, use the Makefile.

## Technical Documentation

Technical documentation (i.e. how to use the C++ API) is available in the doc/ directory.

If not available, then use Doxygen (doxygen.org) to generate the documentation from source code. Use the provided Doxyfile.