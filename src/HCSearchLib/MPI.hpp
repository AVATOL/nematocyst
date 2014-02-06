#ifndef MPI_HPP
#define MPI_HPP

#include <string>

using namespace std;

#ifdef USE_MPI
namespace MPI
{
	class Synchronize
	{
	public:
		// master needs to wait for all slaves to get to this point
		static void masterWait(string slaveBroadcastMsg);
		
		// all slaves need to wait for master to get to this point
		static void slavesWait(string masterBroadcastMsg);
	};
}
#endif

#endif