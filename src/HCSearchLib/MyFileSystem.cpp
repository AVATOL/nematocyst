#include "MyFileSystem.hpp"
#include "Globals.hpp"

namespace MyFileSystem
{
	/**************** FileSystem ****************/

	string FileSystem::normalizeSlashes(string dir)
	{
#if USE_WINDOWS
		string toReplace = "/";
		string replaceWith = "\\";
#else
		string toReplace = "\\";
		string replaceWith = "/";
#endif
		size_t pos = 0;
		while((pos = dir.find(toReplace, pos)) != std::string::npos)
		{
			dir.replace(pos, toReplace.length(), replaceWith);
			pos += replaceWith.length();
		}

		return dir;
	}

	string FileSystem::removeTrailingSlash(string dir)
	{
		if (dir[dir.length()-1] == '\\' || dir[dir.length()-1] == '/')
			dir = dir.substr(0, dir.length()-1);

		return dir;
	}

	string FileSystem::normalizeDirString(string dir)
	{
		return removeTrailingSlash(normalizeSlashes(dir));
	}

	int FileSystem::createFolder(string dir)
	{
		return Executable::execute(HCSearch::Global::settings->cmds->SYSTEM_MKDIR_CMD + " " + dir);
	}

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