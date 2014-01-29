#include <ctime>
#include "Globals.hpp"

namespace HCSearch
{
	/**************** Global ****************/

	namespace Global
	{
		Settings* settings = NULL;
		MyLogger::Logger* log = NULL;
	}

	namespace Rand
	{
		unsigned long g_init[4] = {(int)time(NULL) % 9999, rand() % 9999, (int)time(NULL) % 8888, rand() % 9999};
		unsigned long g_length = 4;
		MTRand_closed unifDist(g_init, g_length);
	}
}