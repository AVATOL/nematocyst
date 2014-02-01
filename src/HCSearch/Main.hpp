#pragma once

#include "MyProgramOptions.hpp"

/*!
 * @brief Demo program.
 *
 * Demonstrates the following:
 * 
 * 1. Loading and unloading the dataset
 * 2. Defining the search space and search procedure
 * 3. Learning H, C, and C with oracle H
 * 4. Saving and loading models
 * 5. Inference with LL, HL, LC and HC searches
 */
void demo(MyProgramOptions::ProgramOptions po);

/*!
 * @brief Run learning and/or inference. 
 * 
 * Schedule depends program options.
 */
void run(MyProgramOptions::ProgramOptions po);

/*!
 * @brief Helper function to set up search space.
 *
 * Define your search space in this function.
 * 
 * The defaults defined here:
 * - Standard CRF features for the heuristic 
 *	and cost feature functions.
 * - IID logistic regression for initial state 
 *	prediction function
 * - Stochastic successor function
 * - Hamming loss function
 */
HCSearch::SearchSpace* setupSearchSpace();

/*!
 * @brief Set up the search space given program options.
 */
HCSearch::SearchSpace* setupSearchSpace(MyProgramOptions::ProgramOptions po);

/*!
 * @brief Set up the search procedure given program options.
 */
HCSearch::ISearchProcedure* setupSearchProcedure(MyProgramOptions::ProgramOptions po);

/*!
 * @brief Convenience function to print schedule.
 */
void printSchedule(MyProgramOptions::ProgramOptions po);