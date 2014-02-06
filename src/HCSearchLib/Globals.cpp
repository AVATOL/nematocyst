#include <iostream>
#include <ctime>
#include <cstdlib>
#include "Globals.hpp"

namespace HCSearch
{
	/**************** Global ****************/

	namespace Global
	{
		Settings* settings = NULL;
	}

	namespace Rand
	{
		unsigned long g_init[4] = {(int)time(NULL) % 9999, rand() % 9999, (int)time(NULL) % 8888, rand() % 9999};
		unsigned long g_length = 4;
		MTRand_closed unifDist(g_init, g_length);
	}

	/**************** Abort ****************/

	void abort()
	{
		abort(1);
	}

	void abort(int errcode)
	{
#ifdef USE_MPI
		cerr << "Process [" << Global::settings->RANK << "] is aborting!" << endl;
		MPI_Abort(MPI_COMM_WORLD, errcode);
#else
		cerr << "Aborting program!" << endl;
		exit(errcode);
#endif
	}
}