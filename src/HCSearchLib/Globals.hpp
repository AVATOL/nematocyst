#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#include "Settings.hpp"
#include "MyLogger.hpp"
#include "mtrand.h"

namespace HCSearch
{
	/**************** Global ****************/

	/*!
	 * @breif Global variables: settings and logging (for debugging)
	 */
	namespace Global
	{
		extern Settings* settings;
		extern MyLogger::Logger* log;
	}

	/*!
	 * @brief Uniform [0, 1] random number generator
	 */
	namespace Rand
	{
		extern unsigned long g_init[4];
		extern unsigned long g_length;
		extern MTRand_closed unifDist;
	}
}

#endif