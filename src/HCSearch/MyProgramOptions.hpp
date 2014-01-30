#pragma once
#include <vector>
#include <string>

using namespace std;

namespace MyProgramOptions
{
	/*!
	 * @brief Stores program options.
	 */
	class ProgramOptions
	{
	public:
		const enum Modes { LEARN_H, LEARN_C, LEARN_C_ORACLE_H, 
			INFER_HC, INFER_HL, INFER_LC, INFER_LL };

		string inputDir;
		string outputDir;

		int timeBound;

		bool demoMode;
		vector< Modes > schedule;

	public:
		ProgramOptions();

		/*!
		 * @brief Parse command line arguments to get program options.
		 */
		static ProgramOptions parseArguments(int argc, char* argv[]);

		/*!
		 * @brief Print command line usage message.
		 */
		static void printUsage(int argc, char* argv[]);
	};
}