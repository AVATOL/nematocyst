#include <iostream>
#include "MyProgramOptions.hpp"
#include "HCSearch.hpp" 

namespace MyProgramOptions
{
	ProgramOptions::ProgramOptions()
	{
		inputDir = "";
		outputDir = "";

		timeBound = 0;

		demoMode = false;
		schedule = vector< Modes >();
	}

	ProgramOptions ProgramOptions::parseArguments(int argc, char* argv[])
	{
		ProgramOptions po;

		if (argc < 4)
		{
			cerr << "Too few arguments!" << endl;
			printUsage();
			HCSearch::abort();
		}
		po.inputDir = argv[1];
		po.outputDir = argv[2];
		po.timeBound = atoi(argv[3]);

		for (int i = 4; i < argc; i++)
		{
			if (strcmp(argv[i], "--demo") == 0)
			{
				po.schedule.clear();
				break;
			}
			else if (strcmp(argv[i], "--learn") == 0)
			{
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "H") == 0)
						po.schedule.push_back(LEARN_H);
					else if (strcmp(argv[i+1], "C") == 0)
						po.schedule.push_back(LEARN_C);
					else if (strcmp(argv[i+1], "COH") == 0)
						po.schedule.push_back(LEARN_C_ORACLE_H);
					else
					{
						po.schedule.push_back(LEARN_H);
						po.schedule.push_back(LEARN_C);
						po.schedule.push_back(LEARN_C_ORACLE_H);
					}
				}
				else
				{
					po.schedule.push_back(LEARN_H);
					po.schedule.push_back(LEARN_C);
					po.schedule.push_back(LEARN_C_ORACLE_H);
				}
			}
			else if (strcmp(argv[i], "--infer") == 0)
			{
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "HC") == 0)
						po.schedule.push_back(INFER_HC);
					else if (strcmp(argv[i+1], "HL") == 0)
						po.schedule.push_back(INFER_HL);
					else if (strcmp(argv[i+1], "LC") == 0)
						po.schedule.push_back(INFER_LC);
					else if (strcmp(argv[i+1], "LL") == 0)
						po.schedule.push_back(INFER_LL);
					else
					{
						po.schedule.push_back(INFER_HC);
						po.schedule.push_back(INFER_HL);
						po.schedule.push_back(INFER_LC);
						po.schedule.push_back(INFER_LL);
					}
				}
				else
				{
					po.schedule.push_back(INFER_HC);
					po.schedule.push_back(INFER_HL);
					po.schedule.push_back(INFER_LC);
					po.schedule.push_back(INFER_LL);
				}
			}
		}

		// demo mode if nothing specified or used --demo flag
		if (po.schedule.empty())
			po.demoMode = true;

		return po;
	}

	void ProgramOptions::printUsage()
	{
		cerr << endl;
		cerr << "Program usage: ./HCSearch INPUT_DIR OUTPUT_DIR TIMEBOUND "
			<< "[--demo] [--learn [H|C|COH]]* [--infer [HC|HL|LC|LL]]* ... [--option=value]" << endl;

		cerr << "Allowed options:" << endl;
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

		cerr << "Notes:" << endl;
		cerr << "* The first three arguments are required. They are the input directory, output directory and time bound." << endl;
		cerr << "* Can use multiple --infer and --learn options in any order to define a schedule. Must come after the mandatory arguments.";
		cerr << endl << endl;
	}
}