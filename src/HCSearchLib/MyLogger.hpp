#ifndef MYLOGGER_HPP
#define MYLOGGER_HPP

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <ctime>

using namespace std;

// Useful macro to shorten log calls
// Usage examples: LOG() << "raw message to log and stdout"
//	or LOG(WARNING) << "auto formatted warning message to log and stderr"
//	or LOG(ERROR, FILE_ONLY) << "auto formatted error message only to log file"
#define LOG MyLogger::Logger::Logger().log

namespace MyLogger
{
	enum LogLevel {DEBUG, INFO, WARNING, ERROR};
	enum OutputMode {FILE_ONLY, STD_ERR, STD_OUT};
	const extern string LogLevelStrings[];

	/*!
	 * Basic logger class for logging errors and debugging.
	 */
	class Logger
	{
		// Globally set settings
	protected:
		static int rank;
		static int numProcesses;
		static ofstream* logstream;
		static LogLevel minLogLevel;

		// For logging
	protected:
		LogLevel logLevel;
		OutputMode outputMode;
		ostringstream os;
		bool embellish;

	public:
		Logger();
		~Logger();

		/*!
		 * Initialize the logger. Must be called before making log calls.
		 * @param rank Rank of process
		 * @param numProcesses Number of processes
		 * @param logPath Path to the log file to open
		 */
		static void initialize(int rank, int numProcesses, string logPath);

		/*!
		 * Set the logging level.
		 * 
		 * Default is INFO. Set to DEBUG to see all messages.
		 * @param level Minimum logging level to display messages: DEBUG|INFO|WARNING|ERROR
		 */
		static void setLogLevel(LogLevel level);

		/*!
		 * Finalize the logger. Must be called before exiting application to release file handler.
		 */
		static void finalize();

		/*!
		 * Write to the log raw stream.
		 */
		ostringstream& log();

		/*!
		 * Write to the log using the specified logging level.
		 * @param level Logging level of message: DEBUG|INFO|WARNING|ERROR
		 */
		ostringstream& log(LogLevel level);

		/*!
		 * Write to the log using the specified logging level 
		 * and writing mode.
		 * @param level Logging level of message: DEBUG|INFO|WARNING|ERROR
		 * @param mode Output mode: FILE_ONLY|STD_ERR|STD_OUT
		 */
		ostringstream& log(LogLevel level, OutputMode mode);

	private:
		double getTimeElapsed();
	};
}

#endif