#pragma once

#include <iostream>
#include "../HCSearchLib/HCSearch.hpp"
#include "MyProgramOptions.hpp"

/*!
 * Demo program. Illustrates the basics.
 * - Loading dataset
 * - Learning the heuristic and cost models
 * - Saving and loading models
 * - Inference on an image using HC search
 * - Unloading dataset
 */
void demo(int timeBound);

/*!
 * Helper to set up the search space.
 */
HCSearch::SearchSpace* setupSearchSpace();