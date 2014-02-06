#include <iostream>
#include "MPI.hpp"
#include "Globals.hpp"

#ifdef USE_MPI
namespace MPI
{
	void Synchronize::masterWait(string slaveBroadcastMsg)
	{
		using namespace HCSearch;

		// receive buffer
		char recvbuff[256];
		const char* SLAVEBROADCASTMSG = slaveBroadcastMsg.c_str();
		const int SLAVEBROADCASTMSG_SIZE = slaveBroadcastMsg.length();
	
		// master needs to wait here until all slaves reaches here
		const int numProcesses = Global::settings->NUM_PROCESSES;
		const int rank = Global::settings->RANK;
		if (rank == 0)
		{
			LOG() << "Master process [" << rank << "] is waiting to get " << slaveBroadcastMsg << " message from all slaves..." << endl;

			// wait until all slaves are done
			bool finish[512];
			finish[0] = true;
			for (int i = 1; i < numProcesses; i++)
			{
				finish[i] = false;
			}

			// master process is now waiting for all slave processes to finish
			while (true)
			{
				if (numProcesses <= 1)
					break;

				int msgFlag = 0;
				MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &msgFlag, Global::settings->MPI_STATUS);

				// if a messsage is available
				if (msgFlag)
				{
					// get the message tag and especially source
					int messageID = (*Global::settings->MPI_STATUS).MPI_TAG;
					int messageSource = (*Global::settings->MPI_STATUS).MPI_SOURCE;
					LOG() << "A message from process [" << messageSource << "]." << endl;

					// receive the message into the buffer and check if it is the correct command
					int ierr = MPI_Recv(recvbuff, SLAVEBROADCASTMSG_SIZE, MPI_CHAR, messageSource, 0, MPI_COMM_WORLD, Global::settings->MPI_STATUS);
					
					// check if correct message
					bool correctMessage = true;
					for (int i = 0; i < SLAVEBROADCASTMSG_SIZE; i++)
					{
						if (recvbuff[i] != slaveBroadcastMsg[i])
						{
							correctMessage = false;
							break;
						}
					}
					
					if (correctMessage)
					{
						LOG() << "Received " << slaveBroadcastMsg << " message from process [" << messageSource << "]." << endl;
						finish[messageSource] = true;

						// test if every process is finished
						bool allFinish = true;
						for (int i = 1; i < numProcesses; i++)
						{
							if (!finish[i])
							{
								LOG() << "Process [" << i << "] is still not here yet..." << endl;
								allFinish = false;
							}
						}
						if (allFinish)
						{
							break;
						}
						LOG() << "Still waiting for all slaves to get here with the master..." << endl;
					}
				}
			}
		
			// MASTER CAN DO STUFF HERE BEFORE SLAVES PROCEED

			LOG() << "Master process [" << rank << "] has continued..." << endl;
		}
		else
		{
			LOG() << "Slave process [" << rank << "] is sending arrival message to master..." << endl;

			// send finish heuristic learning message to master
			int ierr = MPI_Send(const_cast<char*>(SLAVEBROADCASTMSG), SLAVEBROADCASTMSG_SIZE, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

			LOG() << "Slave process [" << rank << "] has continued..." << endl;
		}
	}

	void Synchronize::slavesWait(string masterBroadcastMsg)
	{
		using namespace HCSearch;

		// receive buffer
		char recvbuff[256];
		const char* MASTERBROADCASTMSG = masterBroadcastMsg.c_str();
		const int MASTERBROADCASTMSG_SIZE = masterBroadcastMsg.length();

		// slaves need to wait here until master reaches here
		const int numProcesses = Global::settings->NUM_PROCESSES;
		const int rank = Global::settings->RANK;
		if (rank == 0)
		{
			LOG() << "Master process [" << rank << "] is telling slave processes to continue..." << endl;

			// tell each slave to continue
			for (int j = 1; j < numProcesses; j++)
			{
				int ierr = MPI_Send(const_cast<char*>(MASTERBROADCASTMSG), MASTERBROADCASTMSG_SIZE, MPI_CHAR, j, 0, MPI_COMM_WORLD);
			}

			LOG() << "Master process [" << rank << "] is released..." << endl;
		}
		else
		{
			LOG() << "Slave process [" << rank << "] is waiting for master..." << endl;

			// now wait until master gives the continue signal
			while (true)
			{
				int ierr = MPI_Recv(recvbuff, MASTERBROADCASTMSG_SIZE, MPI_CHAR, 0, 0, MPI_COMM_WORLD, Global::settings->MPI_STATUS);

				// check if correct message
				bool correctMessage = true;
				for (int i = 0; i < MASTERBROADCASTMSG_SIZE; i++)
				{
					if (recvbuff[i] != masterBroadcastMsg.at(i))
					{
						correctMessage = false;
						break;
					}
				}

				if (correctMessage)
				{
					LOG() << "Slave process [" << rank << "] got the " << masterBroadcastMsg << " message." << endl;
					break;
				}
			}

			LOG() << "Slave process [" << rank << "] is released..." << endl;
		}
	}

}
#endif