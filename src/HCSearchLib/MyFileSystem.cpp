#include "MyFileSystem.hpp"

namespace MyFileSystem
{
	/**************** Executable ****************/

	const int Executable::DEFAULT_NUM_RETRIES = 3;

	int Executable::execute(string cmd)
	{
		int retcode = system(cmd.c_str());

		if (retcode != 0)
		{
			cerr << "[Warning] executable cmd '" << cmd << "' returned error code " << retcode << "!" << endl;
		}

		return retcode;
	}

	int Executable::executeRetries(string cmd, int numRetries)
	{
		int retcode = system(cmd.c_str());
		int numRemainingTries = numRetries;
		while (retcode != 0 && numRemainingTries > 0)
		{
			cerr << "[Warning] executable cmd '" << cmd << "' returned error code " << retcode << "!" << endl;
			cerr << "Retrying " << numRemainingTries << " more times..." << endl;
			retcode = system(cmd.c_str());
			numRemainingTries--;
		}

		if (retcode != 0)
		{
			cerr << "[Warning] after " << numRetries << " retries, executable cmd '" << cmd << "' still unable to succeed! Returned error code " << retcode << "!" << endl;
		}

		return retcode;
	}

	int Executable::executeRetries(string cmd)
	{
		return executeRetries(cmd, DEFAULT_NUM_RETRIES);
	}
};