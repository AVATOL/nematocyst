#ifndef MYFILESYSTEM_HPP
#define MYFILESYSTEM_HPP

#include <iostream>
#include <string>
#include <cstdlib>
using namespace std;

namespace MyFileSystem
{
	/**************** FileSystem ****************/

	class FileSystem
	{
	public:
		// Normalize slash direction
		static string normalizeSlashes(string dir);
		
		// Remove trailing slash
		static string removeTrailingSlash(string dir);

		// Normalize slash direction and remove trailing slash
		static string normalizeDirString(string dir);

		// Create a folder
		static int createFolder(string dir);

		// Copy file
		static int copyFile(string src, string dest);

		// Remove file
		static int deleteFile(string path);
	};

	/**************** Executable ****************/

	// For executing command line
	class Executable
	{
	protected:
		// Default number of retries
		static const int DEFAULT_NUM_RETRIES;

	public:
		// Call shell to execute command cmd
		static int execute(string cmd);

		// Call shell to execute command cmd
		// Retry if necessary
		static int executeRetries(string cmd, int numRetries);

		// Call shell to execute command cmd
		// Retry if necessary (default num retries)
		static int executeRetries(string cmd);
	};
}

#endif