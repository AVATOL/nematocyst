#include "MyLogger.hpp"

namespace MyLogger
{
	const string LogLevelStrings[] = {"debug", "info", "WARNING", "ERROR"};

	bool Logger::initialized = false;
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
			if (Logger::initialized)
			{
				(*this->logstream) << os.str();
				(*this->logstream).flush();
			}
			
			// also output
			if (outputMode == STD_ERR)
			{
				cerr << os.str();
				cerr.flush();
			}
			else if (outputMode == STD_OUT)
			{
				cout << os.str();
				cout.flush();
			}
		}
	}

	void Logger::initialize(int rank, int numProcesses, string logPath)
	{
		Logger::rank = rank;
		Logger::numProcesses = numProcesses;
		Logger::minLogLevel = INFO;
		Logger::logstream = new ofstream(logPath.c_str(), std::ios_base::app);
		(*Logger::logstream) << endl << "========== " << getDateTime() << " ==========" << endl << endl;
		Logger::initialized = true;
	}

	void Logger::setLogLevel(LogLevel level)
	{
		Logger::minLogLevel = level;
	}

	void Logger::finalize()
	{
		Logger::initialized = false;
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
		this->embellish = true;

		if (!Logger::initialized)
			return log();

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

	string Logger::getDateTime()
	{
		time_t t = time(0);   // get time now
		struct tm now;
		localtime_s(&now, &t);
		stringstream ss;
		ss << (now.tm_year + 1900) << '-' 
			 << (now.tm_mon + 1) << '-'
			 <<  now.tm_mday << " "
			 << now.tm_hour << ":"
			 << now.tm_min << ":"
			 << now.tm_sec;
		return ss.str();
	}
}