#include <iostream>
#include "MyProgramOptions.hpp"
#include "../HCSearchLib/HCSearch.hpp"

namespace MyProgramOptions
{
	ProgramOptions::ProgramOptions()
	{
		// input and output directories

		baseDir = "";
		inputDir = "";
		outputDir = "";
		splitsFolderName = "splits";
		splitsTrainName = "Train.txt";
		splitsValidName = "Validation.txt";
		splitsTestName = "Test.txt";
		nodesFolderName = "nodes";
		edgesFolderName = "edges";
		edgeFeaturesFolderName = "edgefeatures";

		logsFolderName = "logs";
		modelsFolderName = "models";
		resultsFolderName = "results";
		tempFolderName = "temp";

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
		pruneFeaturesMode = STANDARD_PRUNE;
		initialFunctionMode = LOG_REG;
		successorsMode = STOCHASTIC_CONFIDENCES_NEIGHBORS;
		lossMode = HAMMING;
		pruneMode = NO_PRUNE;

		stochasticCutMode = EDGES;
		beamSize = 1;
		cutParam = 1.0;

		saveAnytimePredictions = true;
		rankLearnerType = HCSearch::SVM_RANK;
		saveFeaturesFiles = false;
		numTrainIterations = 1;
		numTestIterations = 1;
		verboseMode = true;
		uniqueIterId = 0;
		saveOutputMask = false;
		pruneRatio = 0.5;
		badPruneRatio = 1.0;
		useEdgeWeights = false;

		nodeClamp = false;
		edgeClamp = false;
		nodeClampThreshold = 0.9;
		edgeClampPositiveThreshold = 0.9;
		edgeClampNegativeThreshold = 0.1;

		lambda1 = 1.0;
		lambda2 = 1.0;
		lambda3 = 1.0;

		maxCuttingThreshold = 1.0;
		minCuttingThreshold = 0.0;
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
				LOG(ERROR) << "Too few arguments!";
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
			else if (strcmp(argv[i], "--base-path") == 0)
			{
				if (i + 1 != argc)
				{
					po.baseDir = argv[i+1];
				}
			}
			else if (strcmp(argv[i], "--splits-path") == 0)
			{
				if (i + 1 != argc)
				{
					po.splitsFolderName = argv[i+1];
				}
			}
			else if (strcmp(argv[i], "--splits-train-file") == 0)
			{
				if (i + 1 != argc)
				{
					po.splitsTrainName = argv[i+1];
				}
			}
			else if (strcmp(argv[i], "--splits-valid-file") == 0)
			{
				if (i + 1 != argc)
				{
					po.splitsValidName = argv[i+1];
				}
			}
			else if (strcmp(argv[i], "--splits-test-file") == 0)
			{
				if (i + 1 != argc)
				{
					po.splitsTestName = argv[i+1];
				}
			}
			else if (strcmp(argv[i], "--nodes-path") == 0)
			{
				if (i + 1 != argc)
				{
					po.nodesFolderName = argv[i+1];
				}
			}
			else if (strcmp(argv[i], "--edges-path") == 0)
			{
				if (i + 1 != argc)
				{
					po.edgesFolderName = argv[i+1];
				}
			}
			else if (strcmp(argv[i], "--edge-features-path") == 0)
			{
				if (i + 1 != argc)
				{
					po.edgeFeaturesFolderName = argv[i+1];
				}
			}
			else if (strcmp(argv[i], "--logs-path") == 0)
			{
				if (i + 1 != argc)
				{
					po.logsFolderName = argv[i+1];
				}
			}
			else if (strcmp(argv[i], "--models-path") == 0)
			{
				if (i + 1 != argc)
				{
					po.modelsFolderName = argv[i+1];
				}
			}
			else if (strcmp(argv[i], "--results-path") == 0)
			{
				if (i + 1 != argc)
				{
					po.resultsFolderName = argv[i+1];
				}
			}
			else if (strcmp(argv[i], "--temp-path") == 0)
			{
				if (i + 1 != argc)
				{
					po.tempFolderName = argv[i+1];
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
					else if (strcmp(argv[i+1], "P") == 0 || strcmp(argv[i+1], "p") == 0)
						po.schedule.push_back(HCSearch::LEARN_PRUNE);
					else if (strcmp(argv[i+1], "MUTEX") == 0 || strcmp(argv[i+1], "mutex") == 0)
						po.schedule.push_back(HCSearch::DISCOVER_PAIRWISE);
					else if (strcmp(argv[i+1], "ALL") == 0 || strcmp(argv[i+1], "all") == 0)
					{
						po.schedule.push_back(HCSearch::LEARN_H);
						po.schedule.push_back(HCSearch::LEARN_C);
						po.schedule.push_back(HCSearch::LEARN_C_ORACLE_H);
					}
					else
					{
						po.schedule.push_back(HCSearch::LEARN_H);
						po.schedule.push_back(HCSearch::LEARN_C);
					}
				}
				else
				{
					po.schedule.push_back(HCSearch::LEARN_H);
					po.schedule.push_back(HCSearch::LEARN_C);
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
					else if (strcmp(argv[i+1], "ALL") == 0 || strcmp(argv[i+1], "all") == 0)
					{
						po.schedule.push_back(HCSearch::HC);
						po.schedule.push_back(HCSearch::HL);
						po.schedule.push_back(HCSearch::LC);
						po.schedule.push_back(HCSearch::LL);
					}
					else
					{
						po.schedule.push_back(HCSearch::HC);
					}
				}
				else
				{
					po.schedule.push_back(HCSearch::HC);
				}
			}
			else if (strcmp(argv[i], "--ranker") == 0)
			{
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "svmrank") == 0)
						po.rankLearnerType = HCSearch::SVM_RANK;
					else if (strcmp(argv[i+1], "vw") == 0)
						po.rankLearnerType = HCSearch::VW_RANK;
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
			else if (strcmp(argv[i], "--beam-size") == 0)
			{
				if (i + 1 != argc)
				{
					po.beamSize = atoi(argv[i+1]);
					if (po.beamSize <= 0)
					{
						LOG(ERROR) << "Invalid beam size!";
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
					else if (strcmp(argv[i+1], "flipbit-neighbors") == 0)
						po.successorsMode = FLIPBIT_NEIGHBORS;
					else if (strcmp(argv[i+1], "flipbit-confidences-neighbors") == 0)
						po.successorsMode = FLIPBIT_CONFIDENCES_NEIGHBORS;
					else if (strcmp(argv[i+1], "stochastic") == 0)
						po.successorsMode = STOCHASTIC;
					else if (strcmp(argv[i+1], "stochastic-neighbors") == 0)
						po.successorsMode = STOCHASTIC_NEIGHBORS;
					else if (strcmp(argv[i+1], "stochastic-confidences-neighbors") == 0)
						po.successorsMode = STOCHASTIC_CONFIDENCES_NEIGHBORS;
					else if (strcmp(argv[i+1], "cut-schedule") == 0)
						po.successorsMode = CUT_SCHEDULE;
					else if (strcmp(argv[i+1], "cut-schedule-neighbors") == 0)
						po.successorsMode = CUT_SCHEDULE_NEIGHBORS;
					else if (strcmp(argv[i+1], "cut-schedule-confidences-neighbors") == 0)
						po.successorsMode = CUT_SCHEDULE_CONFIDENCES_NEIGHBORS;
					else if (strcmp(argv[i+1], "stochastic-schedule") == 0)
						po.successorsMode = STOCHASTIC_SCHEDULE;
					else if (strcmp(argv[i+1], "stochastic-schedule-neighbors") == 0)
						po.successorsMode = STOCHASTIC_SCHEDULE_NEIGHBORS;
					else if (strcmp(argv[i+1], "stochastic-schedule-confidences-neighbors") == 0)
						po.successorsMode = STOCHASTIC_SCHEDULE_CONFIDENCES_NEIGHBORS;
					else if (strcmp(argv[i+1], "stochastic-constrained") == 0)
					{
						po.successorsMode = STOCHASTIC_CONSTRAINED;
						
						// set clamping to true unless overridden later
						// NOTE: if these arguments were passed before the successor argument, they will be ignored
						po.nodeClamp = true;
						po.edgeClamp = true;
					}
				}
			}
			else if (strcmp(argv[i], "--cut-param") == 0)
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
			else if (strcmp(argv[i], "--save-features") == 0)
			{
				po.saveFeaturesFiles = true;
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "false") == 0)
						po.saveFeaturesFiles = false;
				}
			}
			else if (strcmp(argv[i], "--num-train-iters") == 0)
			{
				if (i + 1 != argc)
				{
					po.numTrainIterations = atoi(argv[i+1]);
					if (po.numTrainIterations <= 0)
					{
						LOG(ERROR) << "Invalid number of iterations!";
						HCSearch::abort();
					}
				}
			}
			else if (strcmp(argv[i], "--num-test-iters") == 0)
			{
				if (i + 1 != argc)
				{
					po.numTestIterations = atoi(argv[i+1]);
					if (po.numTestIterations <= 0)
					{
						LOG(ERROR) << "Invalid number of iterations!";
						HCSearch::abort();
					}
				}
			}
			else if (strcmp(argv[i], "--verbose") == 0)
			{
				po.verboseMode = true;
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "false") == 0)
						po.verboseMode = false;
				}
			}
			else if (strcmp(argv[i], "--cut-mode") == 0)
			{
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "state") == 0)
						po.stochasticCutMode = STATE;
					else if (strcmp(argv[i+1], "edges") == 0)
						po.stochasticCutMode = EDGES;
				}
			}
			else if (strcmp(argv[i], "--unique-iter") == 0)
			{
				if (i + 1 != argc)
				{
					po.uniqueIterId = atoi(argv[i+1]);
					if (po.uniqueIterId < 0)
					{
						LOG(ERROR) << "Unique iteration ID needs to be >= 0";
						HCSearch::abort();
					}
				}
			}
			else if (strcmp(argv[i], "--save-mask") == 0)
			{
				po.saveOutputMask = true;
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "false") == 0)
						po.saveOutputMask = false;
				}
			}
			else if (strcmp(argv[i], "--hfeatures") == 0)
			{
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "standard") == 0)
						po.heuristicFeaturesMode = STANDARD;
					else if (strcmp(argv[i+1], "standard-context") == 0)
						po.heuristicFeaturesMode = STANDARD_CONTEXT;
					//else if (strcmp(argv[i+1], "standard-alt") == 0)
					//	po.heuristicFeaturesMode = STANDARD_ALT;
					else if (strcmp(argv[i+1], "standard-conf") == 0)
						po.heuristicFeaturesMode = STANDARD_CONF;
					else if (strcmp(argv[i+1], "dense-crf") == 0)
						po.heuristicFeaturesMode = DENSE_CRF;
					else if (strcmp(argv[i+1], "unary") == 0)
						po.heuristicFeaturesMode = UNARY;
					else if (strcmp(argv[i+1], "unary-conf") == 0)
						po.heuristicFeaturesMode = UNARY_CONF;
					else if (strcmp(argv[i+1], "standard-pair-counts") == 0)
						po.heuristicFeaturesMode = STANDARD_PAIR_COUNTS;
					else if (strcmp(argv[i+1], "standard-conf-pair-counts") == 0)
						po.heuristicFeaturesMode = STANDARD_CONF_PAIR_COUNTS;
				}
			}
			else if (strcmp(argv[i], "--cfeatures") == 0)
			{
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "standard") == 0)
						po.costFeaturesMode = STANDARD;
					else if (strcmp(argv[i+1], "standard-context") == 0)
						po.costFeaturesMode = STANDARD_CONTEXT;
					//else if (strcmp(argv[i+1], "standard-alt") == 0)
					//	po.costFeaturesMode = STANDARD_ALT;
					else if (strcmp(argv[i+1], "standard-conf") == 0)
						po.costFeaturesMode = STANDARD_CONF;
					else if (strcmp(argv[i+1], "dense-crf") == 0)
						po.costFeaturesMode = DENSE_CRF;
					else if (strcmp(argv[i+1], "unary") == 0)
						po.costFeaturesMode = UNARY;
					else if (strcmp(argv[i+1], "unary-conf") == 0)
						po.costFeaturesMode = UNARY_CONF;
					else if (strcmp(argv[i+1], "standard-pair-counts") == 0)
						po.costFeaturesMode = STANDARD_PAIR_COUNTS;
					else if (strcmp(argv[i+1], "standard-conf-pair-counts") == 0)
						po.costFeaturesMode = STANDARD_CONF_PAIR_COUNTS;
				}
			}
			else if (strcmp(argv[i], "--pfeatures") == 0)
			{
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "standard") == 0)
						po.pruneFeaturesMode = STANDARD;
					else if (strcmp(argv[i+1], "standard-context") == 0)
						po.pruneFeaturesMode = STANDARD_CONTEXT;
					//else if (strcmp(argv[i+1], "standard-alt") == 0)
					//	po.pruneFeaturesMode = STANDARD_ALT;
					else if (strcmp(argv[i+1], "standard-conf") == 0)
						po.pruneFeaturesMode = STANDARD_CONF;
					else if (strcmp(argv[i+1], "dense-crf") == 0)
						po.pruneFeaturesMode = DENSE_CRF;
					else if (strcmp(argv[i+1], "unary") == 0)
						po.pruneFeaturesMode = UNARY;
					else if (strcmp(argv[i+1], "unary-conf") == 0)
						po.pruneFeaturesMode = UNARY_CONF;
					else if (strcmp(argv[i+1], "standard-pair-counts") == 0)
						po.pruneFeaturesMode = STANDARD_PAIR_COUNTS;
					else if (strcmp(argv[i+1], "standard-conf-pair-counts") == 0)
						po.pruneFeaturesMode = STANDARD_CONF_PAIR_COUNTS;
					else if (strcmp(argv[i+1], "standard-prune") == 0)
						po.pruneFeaturesMode = STANDARD_PRUNE;
				}
			}
			else if (strcmp(argv[i], "--loss") == 0)
			{
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "hamming") == 0)
						po.lossMode = HAMMING;
					else if (strcmp(argv[i+1], "pixel-hamming") == 0)
						po.lossMode = PIXEL_HAMMING;
				}
			}
			else if (strcmp(argv[i], "--prune") == 0)
			{
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "none") == 0)
						po.pruneMode = NO_PRUNE;
					else if (strcmp(argv[i+1], "ranker") == 0)
						po.pruneMode = RANKER_PRUNE;
					else if (strcmp(argv[i+1], "oracle") == 0)
						po.pruneMode = ORACLE_PRUNE;
					else if (strcmp(argv[i+1], "simulated") == 0)
						po.pruneMode = SIMULATED_RANKER_PRUNE;
				}
			}
			else if (strcmp(argv[i], "--prune-ratio") == 0)
			{
				if (i + 1 != argc)
				{
					po.pruneRatio = atof(argv[i+1]);
					if (po.pruneRatio < 0 || po.pruneRatio >= 1)
					{
						LOG(ERROR) << "Prune ratio needs to be between 0 and 1";
						HCSearch::abort();
					}
				}
			}
			else if (strcmp(argv[i], "--prune-bad-ratio") == 0)
			{
				if (i + 1 != argc)
				{
					po.badPruneRatio = atof(argv[i+1]);
					if (po.badPruneRatio < 0 || po.badPruneRatio > 1)
					{
						LOG(ERROR) << "Prune ratio needs to be between 0 and 1";
						HCSearch::abort();
					}
				}
			}
			else if (strcmp(argv[i], "--node-clamp") == 0)
			{
				po.nodeClamp = true;
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "false") == 0)
						po.nodeClamp = false;
				}
			}
			else if (strcmp(argv[i], "--edge-clamp") == 0)
			{
				po.edgeClamp = true;
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "false") == 0)
						po.edgeClamp = false;
				}
			}
			else if (strcmp(argv[i], "--node-clamp-threshold") == 0)
			{
				if (i + 1 != argc)
				{
					po.nodeClampThreshold = atof(argv[i+1]);
					if (po.nodeClampThreshold < 0 || po.nodeClampThreshold > 1)
					{
						LOG(ERROR) << "Clamp ratio needs to be between 0 and 1";
						HCSearch::abort();
					}
				}
			}
			else if (strcmp(argv[i], "--edge-clamp-positive-threshold") == 0)
			{
				if (i + 1 != argc)
				{
					po.edgeClampPositiveThreshold = atof(argv[i+1]);
					if (po.edgeClampPositiveThreshold < 0 || po.edgeClampPositiveThreshold > 1)
					{
						LOG(ERROR) << "Clamp ratio needs to be between 0 and 1";
						HCSearch::abort();
					}
				}
			}
			else if (strcmp(argv[i], "--edge-clamp-negative-threshold") == 0)
			{
				if (i + 1 != argc)
				{
					po.edgeClampNegativeThreshold = atof(argv[i+1]);
					if (po.edgeClampNegativeThreshold < 0 || po.edgeClampNegativeThreshold > 1)
					{
						LOG(ERROR) << "Clamp ratio needs to be between 0 and 1";
						HCSearch::abort();
					}
				}
			}
			else if (strcmp(argv[i], "--lambda1") == 0)
			{
				if (i + 1 != argc)
				{
					po.lambda1 = atof(argv[i+1]);
				}
			}
			else if (strcmp(argv[i], "--lambda2") == 0)
			{
				if (i + 1 != argc)
				{
					po.lambda2 = atof(argv[i+1]);
				}
			}
			else if (strcmp(argv[i], "--lambda3") == 0)
			{
				if (i + 1 != argc)
				{
					po.lambda3 = atof(argv[i+1]);
				}
			}
			else if (strcmp(argv[i], "--max-cutting-threshold") == 0)
			{
				if (i + 1 != argc)
				{
					po.maxCuttingThreshold = atof(argv[i+1]);
					if (po.maxCuttingThreshold < 0 || po.maxCuttingThreshold > 1)
					{
						LOG(ERROR) << "Threshold needs to be between 0 and 1";
						HCSearch::abort();
					}
				}
			}
			else if (strcmp(argv[i], "--min-cutting-threshold") == 0)
			{
				if (i + 1 != argc)
				{
					po.minCuttingThreshold = atof(argv[i+1]);
					if (po.minCuttingThreshold < 0 || po.minCuttingThreshold > 1)
					{
						LOG(ERROR) << "Threshold needs to be between 0 and 1";
						HCSearch::abort();
					}
				}
			}
			else if (strcmp(argv[i], "--use-edge-weights") == 0)
			{
				po.useEdgeWeights = true;
				if (i + 1 != argc)
				{
					if (strcmp(argv[i+1], "false") == 0)
						po.useEdgeWeights = false;
				}
			}
			else
			{
				string argvi = argv[i];
				if (argvi.find("--") == 0)
					LOG(WARNING) << "IGNORING UNKNOWN PROGRAM OPTION: " << argv[i];
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
			<< "[--learn (H|C|COH)]* [--infer (HC|HL|LC|LL)]* ... [--option=value]" << endl;

		cerr << "Main options:" << endl;
		cerr << "\t--help\t\t" << ": produce help message" << endl;
		cerr << "\t--demo\t\t" << ": run the demo program (ignores --learn and --infer)" << endl;
		cerr << "\t--learn arg\t" << ": learning" << endl;
		cerr << "\t\t\t\tH: learn heuristic" << endl;
		cerr << "\t\t\t\tC: learn cost" << endl;
		cerr << "\t\t\t\tCOH: learn cost with oracle H" << endl;
		cerr << "\t\t\t\tP: learn prune function" << endl;
		cerr << "\t\t\t\tALL: short-hand for H, C, COH" << endl;
		cerr << "\t\t\t\t(none): short-hand for H, C" << endl;
		cerr << "\t--infer arg\t" << ": inference" << endl;
		cerr << "\t\t\t\tHC: learned heuristic and cost" << endl;
		cerr << "\t\t\t\tHL: learned heuristic and oracle cost" << endl;
		cerr << "\t\t\t\tLC: oracle heuristic and learned cost" << endl;
		cerr << "\t\t\t\tLL: oracle heuristic and cost" << endl;
		cerr << "\t\t\t\tALL: short-hand for HC, HL, LC, LL" << endl;
		cerr << "\t\t\t\t(none): short-hand for HC" << endl;
		cerr << endl;

		cerr << "Advanced options:" << endl;
		cerr << "\t--anytime arg\t\t\t" << ": turn on saving anytime predictions if true" << endl;
		cerr << "\t--base-path arg\t\t\t" << ": path to this program (default is pwd)" << endl;
		cerr << "\t--beam-size arg\t\t\t" << ": beam size for beam search" << endl;
		cerr << "\t--cut-mode arg\t\t\t" << ": edges|state (cut edges by edges independently or by state)" << endl;
		cerr << "\t--cut-param arg\t\t\t" << ": temperature parameter for stochastic cuts" << endl;
		cerr << "\t--edge-clamp arg\t" << ": clamp edges if true" << endl;
		cerr << "\t--edge-clamp-positive-threshold arg\t" << ": edge clamp positive threshold" << endl;
		cerr << "\t--edge-clamp-negative-threshold arg\t" << ": edge clamp negative threshold" << endl;
		cerr << "\t--edges-path arg\t" << ": edges folder name" << endl;
		cerr << "\t--edge-features-path arg\t" << ": edge features folder name" << endl;
		cerr << "\t--hfeatures arg\t\t\t" << ": standard|standard-context|standard-conf|unary|unary-conf|"
			"standard-pair-counts|standard-conf-pair-counts|dense-crf" << endl;
		cerr << "\t--cfeatures arg\t\t\t" << ": standard|standard-context|standard-conf|unary|unary-conf|"
			"standard-pair-counts|standard-conf-pair-counts|dense-crf" << endl;
		cerr << "\t--pfeatures arg\t\t" << ": standard|standard-context|standard-conf|unary|unary-conf|"
			"standard-pair-counts|standard-conf-pair-counts|dense-crf|standard-prune" << endl;
		cerr << "\t--logs-path arg\t" << ": logs folder name" << endl;
		cerr << "\t--max-cutting-threshold arg\t" << ": maximum threshold for deciding to cut edges" << endl;
		cerr << "\t--min-cutting-threshold arg\t" << ": minimum threshold for deciding to cut edges" << endl;
		cerr << "\t--models-path arg\t" << ": models folder name" << endl;
		cerr << "\t--node-clamp arg\t" << ": clamp nodes if true" << endl;
		cerr << "\t--node-clamp-threshold arg\t" << ": node clamp threshold" << endl;
		cerr << "\t--nodes-path arg\t" << ": nodes folder name" << endl;
		cerr << "\t--num-test-iters arg\t" << ": number of test iterations" << endl;
		cerr << "\t--num-train-iters arg\t" << ": number of training iterations" << endl;
		cerr << "\t--ranker arg\t\t\t" << ": svmrank|vw" << endl;
		cerr << "\t--loss arg\t\t\t\t" << ": hamming|pixel-hamming" << endl;
		cerr << "\t--prune arg\t\t" << ": none|ranker|oracle|simulated" << endl;
		cerr << "\t--prune-ratio arg\t\t" << ": fraction of candidates to prune" << endl;
		cerr << "\t--prune-bad-ratio arg\t\t" << ": fraction of bad candidates to prune for oracle pruner" << endl;
		cerr << "\t--results-path arg\t" << ": results folder name" << endl;
		cerr << "\t--save-features arg\t\t" << ": save rank features during learning if true" << endl;
		cerr << "\t--save-mask arg\t\t\t" << ": save final prediction label masks if true" << endl;
		cerr << "\t--search arg\t\t\t" << ": greedy|breadthbeam|bestbeam" << endl;
		cerr << "\t--splits-path arg\t\t" << ": specify alternate path to splits folder" << endl;
		cerr << "\t--splits-train-file arg\t" << ": specify alternate file name to train file" << endl;
		cerr << "\t--splits-valid-file arg\t" << ": specify alternate file name to validation file" << endl;
		cerr << "\t--splits-test-file arg\t" << ": specify alternate file name to test file" << endl;
		cerr << "\t--successor arg\t\t\t" << ": flipbit|flipbit-neighbors|flipbit-confidences-neighbors|"
			<< "stochastic|stochastic-neighbors|stochastic-confidences-neighbors|"
			<< "cut-schedule|cut-schedule-neighbors|cut-schedule-confidences-neighbors"
			<< "stochastic-schedule|stochastic-schedule-neighbors|stochastic-schedule-confidences-neighbors|stochastic-constrained" << endl;
		cerr << "\t--temp-path arg\t" << ": temp folder name" << endl;
		cerr << "\t--use-edge-weights arg\t\t" << ": use edge weights if true (must provide --edges-path)" << endl;
		cerr << "\t--unique-iter arg\t\t" << ": unique iteration ID (num-test-iters needs to be 1)" << endl;
		cerr << "\t--verbose arg\t\t\t" << ": turn on verbose output if true" << endl;
		cerr << endl;

		cerr << "Notes:" << endl;
		cerr << "* The first three arguments are required. They are the input directory, output directory and time bound." << endl;
		cerr << "* Can use multiple --infer and --learn options in any order to define a schedule. Must come after the mandatory arguments.";
		cerr << endl << endl;
	}
}