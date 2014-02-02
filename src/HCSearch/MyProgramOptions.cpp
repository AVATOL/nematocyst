#include <iostream>
#include "MyProgramOptions.hpp"
#include "../HCSearchLib/HCSearch.hpp"

namespace MyProgramOptions
{
	ProgramOptions::ProgramOptions()
	{
		// input and output directories

		inputDir = "";
		outputDir = "";
		splitsFolderName = "splits";

		// time bound

		timeBound = 0;

		// schedule or demo

		printUsageMode = false;
		demoMode = false;
		schedule = vector< HCSearch::SearchType >();

		// options

		searchProcedureMode = GREEDY;
		heuristicFeaturesMode = STANDARD;
		costFeaturesMode = STANDARD;
		initialFunctionMode = LOG_REG;
		successorsMode = STOCHASTIC;
		lossMode = HAMMING;

		stochasticCutMode = STATE;
		beamSize = 1;
		cutParam = 1.0;

		saveAnytimePredictions = true;
		rankLearnerType = HCSearch::SVM_RANK;
	}

	ProgramOptions ProgramOptions::parseArguments(int argc, char* argv[])
	{
		ProgramOptions po;

		if ((argc >= 2 && strcmp(argv[1], "--help") == 0) || 
			(argc >= 3 && strcmp(argv[2], "--help") == 0) || 
			(argc >= 4 && strcmp(argv[3], "--help") == 0))
		{
			po.printUsageMode = true;
			return po;
		}

		if (argc < 4)
		{
			if (HCSearch::Global::settings->RANK == 0)
			{
				cerr << "Too few arguments!" << endl;
				printUsage();
			}
			HCSearch::abort();
		}
		po.inputDir = argv[1];
		po.outputDir = argv[2];
		po.timeBound = atoi(argv[3]);

		for (int i = 4; i < argc; i++)
		{
			if (strcmp(argv[i], "--help") == 0)
			{
				po.printUsageMode = true;
				break;
			}
			else if (strcmp(argv[i], "--demo") == 0)
			{
				po.demoMode = true;
			}
			else if (strcmp(argv[i], "--splitspath") == 0)
			{
				if (i + 1 != argc)
				{
					po.splitsFolderName = argv[i+1];
				}
			}
			else if (strcmp(argv[i], "--learn") == 0)
			{
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "H") == 0 || strcmp(argv[i+1], "h") == 0)
						po.schedule.push_back(HCSearch::LEARN_H);
					else if (strcmp(argv[i+1], "C") == 0 || strcmp(argv[i+1], "c") == 0)
						po.schedule.push_back(HCSearch::LEARN_C);
					else if (strcmp(argv[i+1], "COH") == 0 || strcmp(argv[i+1], "coh") == 0)
						po.schedule.push_back(HCSearch::LEARN_C_ORACLE_H);
					else
					{
						po.schedule.push_back(HCSearch::LEARN_H);
						po.schedule.push_back(HCSearch::LEARN_C);
						po.schedule.push_back(HCSearch::LEARN_C_ORACLE_H);
					}
				}
				else
				{
					po.schedule.push_back(HCSearch::LEARN_H);
					po.schedule.push_back(HCSearch::LEARN_C);
					po.schedule.push_back(HCSearch::LEARN_C_ORACLE_H);
				}
			}
			else if (strcmp(argv[i], "--infer") == 0)
			{
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "HC") == 0 || strcmp(argv[i+1], "hc") == 0)
						po.schedule.push_back(HCSearch::HC);
					else if (strcmp(argv[i+1], "HL") == 0 || strcmp(argv[i+1], "hl") == 0)
						po.schedule.push_back(HCSearch::HL);
					else if (strcmp(argv[i+1], "LC") == 0 || strcmp(argv[i+1], "lc") == 0)
						po.schedule.push_back(HCSearch::LC);
					else if (strcmp(argv[i+1], "LL") == 0 || strcmp(argv[i+1], "ll") == 0)
						po.schedule.push_back(HCSearch::LL);
					else
					{
						po.schedule.push_back(HCSearch::HC);
						po.schedule.push_back(HCSearch::HL);
						po.schedule.push_back(HCSearch::LC);
						po.schedule.push_back(HCSearch::LL);
					}
				}
				else
				{
					po.schedule.push_back(HCSearch::HC);
					po.schedule.push_back(HCSearch::HL);
					po.schedule.push_back(HCSearch::LC);
					po.schedule.push_back(HCSearch::LL);
				}
			}
			else if (strcmp(argv[i], "--learner") == 0)
			{
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "svmrank") == 0)
						po.rankLearnerType = HCSearch::SVM_RANK;
					else if (strcmp(argv[i+1], "online") == 0)
						po.rankLearnerType = HCSearch::ONLINE_RANK;
				}
			}
			else if (strcmp(argv[i], "--search") == 0)
			{
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "greedy") == 0)
						po.searchProcedureMode = GREEDY;
					else if (strcmp(argv[i+1], "beam") == 0)
						po.searchProcedureMode = BREADTH_BEAM;
					else if (strcmp(argv[i+1], "breadthbeam") == 0)
						po.searchProcedureMode = BREADTH_BEAM;
					else if (strcmp(argv[i+1], "bestbeam") == 0)
						po.searchProcedureMode = BEST_BEAM;
				}
			}
			else if (strcmp(argv[i], "--beamsize") == 0)
			{
				if (i + 1 != argc)
				{
					po.beamSize = atoi(argv[i+1]);
					if (po.beamSize <= 0)
					{
						cerr << "Invalid beam size!" << endl;
						HCSearch::abort();
					}
				}
			}
			else if (strcmp(argv[i], "--successor") == 0)
			{
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "flipbit") == 0)
						po.successorsMode = FLIPBIT;
					else if (strcmp(argv[i+1], "stochastic") == 0)
						po.successorsMode = STOCHASTIC;
				}
			}
			else if (strcmp(argv[i], "--cutparam") == 0)
			{
				if (i + 1 != argc)
				{
					po.cutParam = atof(argv[i+1]);
				}
			}
			else if (strcmp(argv[i], "--anytime") == 0)
			{
				po.saveAnytimePredictions = true;
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "false") == 0)
						po.saveAnytimePredictions = false;
				}
			}
		}

		// demo mode if nothing specified or used --demo flag
		if (po.schedule.empty() && !po.demoMode)
			po.demoMode = true;
		else if (po.demoMode)
			po.schedule.clear();

		return po;
	}

	void ProgramOptions::printUsage()
	{
		cerr << endl;
		cerr << "Program usage: ./HCSearch INPUT_DIR OUTPUT_DIR TIMEBOUND "
			<< "[--learn [H|C|COH]]* [--infer [HC|HL|LC|LL]]* ... [--option=value]" << endl;

		cerr << "Main options:" << endl;
		cerr << "\t--help\t\t" << ": produce help message" << endl;
		cerr << "\t--demo\t\t" << ": run the demo program (ignores --learn and --infer)" << endl;
		cerr << "\t--learn arg\t" << ": learning" << endl;
		cerr << "\t\t\t\tH: learn heuristic" << endl;
		cerr << "\t\t\t\tC: learn cost" << endl;
		cerr << "\t\t\t\tCOH: learn cost with oracle H" << endl;
		cerr << "\t\t\t\t(none): short-hand for H, C, COH" << endl;
		cerr << "\t--infer arg\t" << ": inference" << endl;
		cerr << "\t\t\t\tHC: learned heuristic and cost" << endl;
		cerr << "\t\t\t\tHL: learned heuristic and oracle cost" << endl;
		cerr << "\t\t\t\tLC: oracle heuristic and learned cost" << endl;
		cerr << "\t\t\t\tLL: oracle heuristic and cost" << endl;
		cerr << "\t\t\t\t(none): short-hand for HC, HL, LC, LL" << endl;
		cerr << endl;

		cerr << "Advanced options:" << endl;
		cerr << "\t--anytime arg\t" << ": turn off saving anytime predictions if false" << endl;
		cerr << endl;

		cerr << "Notes:" << endl;
		cerr << "* The first three arguments are required. They are the input directory, output directory and time bound." << endl;
		cerr << "* Can use multiple --infer and --learn options in any order to define a schedule. Must come after the mandatory arguments.";
		cerr << endl << endl;
	}
}