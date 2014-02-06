#include "MyLogger.hpp"

namespace MyLogger
{
	const string LogLevelStrings[] = {"debug", "info", "WARNING", "ERROR"};

	int Logger::rank = -1;
	int Logger::numProcesses = 0;
	ofstream* Logger::logstream = NULL;
	LogLevel Logger::minLogLevel = INFO;

	Logger::Logger()
	{
		this->embellish = false;
	}

	Logger::~Logger()
	{
		if (this->logLevel >= Logger::minLogLevel)
		{
			if (embellish)
				os << endl;

			// to file
			(*this->logstream) << os.str();
				
			// also output
			if (outputMode == STD_ERR)
			{
				cerr << os.str();
				fflush(stderr);
			}
			else if (outputMode == STD_OUT)
			{
				cout << os.str();
				fflush(stdout);
			}
		}
	}

	void Logger::initialize(int rank, int numProcesses, string logPath)
	{
		Logger::rank = rank;
		Logger::numProcesses = numProcesses;
		Logger::minLogLevel = INFO;
		Logger::logstream = new ofstream(logPath.c_str(), std::ios_base::app);

		(*Logger::logstream) << "==========" << endl;
	}

	void Logger::setLogLevel(LogLevel level)
	{
		Logger::minLogLevel = level;
	}

	void Logger::finalize()
	{
		Logger::logstream->close();
		delete Logger::logstream;
	}

	ostringstream& Logger::log()
	{
		this->logLevel = INFO;
		this->outputMode = STD_OUT;
		return os;
	}

	ostringstream& Logger::log(LogLevel level)
	{
		if (level >= WARNING)
			return log(level, STD_ERR);
		else
			return log(level, STD_OUT);
	}

	ostringstream& Logger::log(LogLevel level, OutputMode mode)
	{
		if (Logger::rank == -1)
			return os;

		this->embellish = true;
		this->logLevel = level;
		this->outputMode = mode;
		os << "[" << rank << "/" << numProcesses << "] "
			<< "[" << getTimeElapsed() << "] "
			<< "(" << LogLevelStrings[level] << ") ";
		return os;
	}

	double Logger::getTimeElapsed()
	{
		return (double)clock()/CLOCKS_PER_SEC;
	}
}