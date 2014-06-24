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

	int FileSystem::copyFile(string src, string dest)
	{
		if (src.compare(dest) == 0)
		{
			LOG(WARNING) << "copy src and dest the same." << endl;
			return 0;
		}

		return Executable::execute(HCSearch::Global::settings->cmds->SYSTEM_COPY_CMD + " " + src + " " + dest);
	}

	int FileSystem::deleteFile(string path)
	{
		return Executable::execute(HCSearch::Global::settings->cmds->SYSTEM_RM_CMD + " " + path);
	}

	bool FileSystem::checkFileExists(string path)
	{
		if (FILE *file = fopen(path.c_str(), "r")) {
			fclose(file);
			return true;
		} else {
			return false;
		}
	}

	/**************** Executable ****************/

	const int Executable::DEFAULT_NUM_RETRIES = 3;

	int Executable::execute(string cmd)
	{
		int retcode = system(cmd.c_str());

		if (retcode != 0)
		{
			LOG(WARNING) << "executable cmd '" << cmd << "' returned error code " << retcode << "!";
		}

		return retcode;
	}

	int Executable::executeRetries(string cmd, int numRetries)
	{
		int retcode = system(cmd.c_str());
		int numRemainingTries = numRetries;
		while (retcode != 0 && numRemainingTries > 0)
		{
			LOG(WARNING) << "executable cmd '" << cmd << "' returned error code " << retcode << "! " 
				<< "Retrying " << numRemainingTries << " more times...";
			retcode = system(cmd.c_str());
			numRemainingTries--;
		}

		if (retcode != 0)
		{
			LOG(WARNING) << "after " << numRetries << " retries, executable cmd '" << cmd << "' still unable to succeed! Returned error code " << retcode << "!";
		}

		return retcode;
	}

	int Executable::executeRetries(string cmd)
	{
		return executeRetries(cmd, DEFAULT_NUM_RETRIES);
	}

	int Executable::executeRetriesFatal(string cmd, int numRetries)
	{
		int retcode = system(cmd.c_str());
		int numRemainingTries = numRetries;
		while (retcode != 0 && numRemainingTries > 0)
		{
			LOG(WARNING) << "executable cmd '" << cmd << "' returned error code " << retcode << "! " 
				<< "Retrying " << numRemainingTries << " more times...";
			retcode = system(cmd.c_str());
			numRemainingTries--;
		}

		if (retcode != 0)
		{
			LOG(ERROR) << "after " << numRetries << " retries, executable cmd '" << cmd << "' still unable to succeed! Returned error code " << retcode << "!";
			LOG(ERROR) << "aborting since all attempts failed!";
			HCSearch::abort();
		}

		return retcode;
	}

	int Executable::executeRetriesFatal(string cmd)
	{
		return executeRetriesFatal(cmd, DEFAULT_NUM_RETRIES);
	}
};