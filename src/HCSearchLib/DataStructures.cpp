#include <vector>
#include <fstream>
#include <iostream>
#include "DataStructures.hpp"
#include "Globals.hpp"
#include "MyFileSystem.hpp"

using namespace std;

namespace HCSearch
{
	/**************** Constants ****************/

	const string SearchTypeStrings[] = {"ll", "hl", "lc", "hc", "learnh", "learnc", "learncoracle"};
	const string DatasetTypeStrings[] = {"test", "train", "validation"};

	/**************** Features and Labelings ****************/

	ImgFeatures::ImgFeatures()
	{
		this->filename = "";
		this->segmentsAvailable = false;
	}

	ImgFeatures::~ImgFeatures()
	{
	}

	int ImgFeatures::getFeatureDim()
	{
		return this->graph.nodesData.cols();
	}

	int ImgFeatures::getNumNodes()
	{
		return this->graph.nodesData.rows();
	}

	double ImgFeatures::getFeature(int node, int featIndex)
	{
		return this->graph.nodesData(node, featIndex);
	}

	string ImgFeatures::getFileName()
	{
		return this->filename;
	}

	ImgLabeling::ImgLabeling()
	{
		this->confidencesAvailable = false;
		this->stochasticCutsAvailable = false;
	}

	ImgLabeling::~ImgLabeling()
	{
	}

	int ImgLabeling::getNumNodes()
	{
		return this->graph.nodesData.size();
	}

	int ImgLabeling::getLabel(int node)
	{
		return this->graph.nodesData(node);
	}

	set<int> ImgLabeling::getNeighborLabels(int node)
	{
		set<int> labels;
		if (hasNeighbors(node))
		{
			for (set<int>::iterator it = this->graph.adjList[node].begin();
				it != this->graph.adjList[node].end(); ++it)
			{
				labels.insert(getLabel(*it));
			}
		}
		return labels;
	}

	set<int> ImgLabeling::getNeighbors(int node)
	{
		if (hasNeighbors(node))
			return this->graph.adjList[node];
		else
			return set<int>();
	}

	bool ImgLabeling::hasNeighbors(int node)
	{
		return this->graph.adjList.count(node) != 0;
	}

	/**************** Rank Features ****************/

	RankFeatures::RankFeatures()
	{
	}

	RankFeatures::RankFeatures(VectorXd features)
	{
		this->data = features;
	}

	RankFeatures::~RankFeatures()
	{
	}

	/**************** SVM-Rank Model ****************/

	SVMRankModel::SVMRankModel()
	{
		this->initialized = false;
		this->learningMode = false;
	}

	SVMRankModel::SVMRankModel(string fileName)
	{
		load(fileName);
	}
	
	double SVMRankModel::rank(RankFeatures features)
	{
		if (!this->initialized)
		{
			cerr << "[Error] svm ranker not initialized for ranking" << endl;
			exit(1);
		}

		return vectorDot(getWeights(), features.data);
	}

	RankerType SVMRankModel::rankerType()
	{
		return SVM_RANK;
	}

	void SVMRankModel::load(string fileName)
	{
		this->weights = parseModelFile(fileName);
		this->initialized = true;
	}

	void SVMRankModel::save(string fileName)
	{
		writeModelFile(fileName, this->weights);
	}
	
	VectorXd SVMRankModel::getWeights()
	{
		if (!this->initialized)
		{
			cerr << "[Error] svm ranker not initialized for getting weights" << endl;
			exit(1);
		}

		return this->weights;
	}

	void SVMRankModel::startTraining(string featuresFileName)
	{
		this->learningMode = true;
		this->qid = 1;
		this->rankingFile = new ofstream(featuresFileName.c_str());
		this->rankingFileName = featuresFileName;
	}

	void SVMRankModel::addTrainingExamples(vector< RankFeatures >& betterSet, vector< RankFeatures >& worseSet)
	{
		cout << "Training with " << betterSet.size() << " best examples and " << worseSet.size() << " worst examples..." << endl;

		// good examples
		for (vector< RankFeatures >::iterator it = betterSet.begin(); it != betterSet.end(); ++it)
		{
			RankFeatures better = *it;
			(*this->rankingFile) << vector2svmrank(better, 1, this->qid) << endl;

			// bad examples
			for (vector< RankFeatures >::iterator it2 = worseSet.begin(); it2 != worseSet.end(); ++it2)
			{
				RankFeatures worse = *it2;
				(*this->rankingFile) << vector2svmrank(worse, 2, this->qid) << endl;
			}

			// increment qid
			this->qid++;
		}
	}

	void SVMRankModel::finishTraining(string modelFileName, SearchType searchType)
	{
		if (searchType != LEARN_H && searchType != LEARN_C && searchType != LEARN_C_ORACLE_H)
		{
			cerr << "[Error] invalid search type for training." << endl;
			abort();
		}

		if (qid <= 1)
		{
			cerr << "[Error] no training data available for learning!" << endl;
			return;
		}

		// close ranking file
		this->rankingFile->close();
		delete this->rankingFile;

#ifdef USE_MPI
		string STARTMSG;
		string ENDMSG;
		string featuresFileBase;
		if (searchType == LEARN_H)
		{
			STARTMSG = "MERGEHSTART";
			ENDMSG = "MERGEHEND";
			featuresFileBase = Global::settings->paths->OUTPUT_HEURISTIC_FEATURES_FILE_BASE;
		}
		else if (searchType == LEARN_C)
		{
			STARTMSG = "MERGECSTART";
			ENDMSG = "MERGECEND";
			featuresFileBase = Global::settings->paths->OUTPUT_COST_H_FEATURES_FILE_BASE;
		}
		else if (searchType == LEARN_C_ORACLE_H)
		{
			STARTMSG = "MERGECOHSTART";
			ENDMSG = "MERGECOHEND";
			featuresFileBase = Global::settings->paths->OUTPUT_COST_ORACLE_H_FEATURES_FILE_BASE;
		}

		MPI::Synchronize::masterWait(STARTMSG);

		// merge step
		if (Global::settings->RANK == 0)
		{
			this->qid = mergeRankingFiles(featuresFileBase, Global::settings->NUM_PROCESSES, this->qid-1);
		}
#endif

		// training step
		if (Global::settings->RANK == 0)
		{
			// compute C
			double C = 1.0 * (this->qid-1);

			// call SVM-Rank
			stringstream ssLearn;
			ssLearn << Global::settings->cmds->SVMRANK_LEARN_CMD << " -c " << C << " " 
				<< this->rankingFileName << " " << modelFileName;
			MyFileSystem::Executable::executeRetries(ssLearn.str());
		}

#ifdef USE_MPI
		MPI::Synchronize::slavesWait(ENDMSG);
#endif

		// no longer learning
		this->learningMode = false;

		// load weights into model and initialize
		load(Global::settings->paths->OUTPUT_HEURISTIC_MODEL_FILE);

		cout << endl;
	}

	void SVMRankModel::cancelTraining()
	{
		// close ranking file
		this->rankingFile->close();
		delete this->rankingFile;

		// no longer learning
		this->learningMode = false;
	}

	VectorXd SVMRankModel::parseModelFile(string fileName)
	{
		string line;
		VectorXd weights;
		vector<int> indices;
		vector<double> values;

		ifstream fh(fileName.c_str());
		if (fh.is_open())
		{
			int lineNum = 0;
			while (fh.good())
			{
				lineNum++;
				getline(fh, line);
				if (lineNum < 12)
					continue;

				if (lineNum == 12)
				{
					istringstream iss(line);

					string token;
					while (getline(iss, token, ' '))
					{
						if (token.find(':') == std::string::npos)
							continue;

						istringstream isstoken(token);
						string sIndex;
						string sValue;
						getline(isstoken, sIndex, ':');
						getline(isstoken, sValue, ':');

						int index = atoi(sIndex.c_str());
						double value = atof(sValue.c_str());

						indices.push_back(index);
						values.push_back(value);
					}
				}
			}
			fh.close();
		}
		else
		{
			cerr << "[Error] cannot open model file for reading weights!!" << endl;
			exit(1);
		}

		weights = VectorXd::Zero(indices.back());
		int valuesSize = values.size();
		if (valuesSize == 0)
		{
			cerr << "[Error] assigning empty weights from '" + fileName + "'!" << endl;
			exit(1);
		}

		for (int i = 0; i < valuesSize; i++)
		{
			int ind = indices[i]-1;
			weights(ind) = values[i];
		}

		return weights;
	}

	string SVMRankModel::vector2svmrank(RankFeatures features, int target, int qid)
	{
		stringstream ss("");
		stringstream sparse("");

		VectorXd vector = features.data;

		int nonZeroCounts = 0;
		for (int i = 0; i < vector.size(); i++)
		{
			if (vector(i) != 0)
			{
				ss << i+1 << ":" << vector(i) << " ";
				nonZeroCounts++;
			}
		}
		sparse << target << " qid:" << qid << " " << ss.str();

		return sparse.str();
	}

	void SVMRankModel::writeModelFile(string fileName, const VectorXd& weights)
	{
		ofstream fh(fileName.c_str());
		if (fh.is_open())
		{
			// write num to file
			fh << "SVM-light - generated from HC-Search" << endl;

			// jump to line 12
			for (int i = 0; i < 10; i++)
				fh << "#" << endl;

			// write weights to file
			fh << "1 ";
			const int weightsLength = weights.size();
			for (int i = 0; i < weightsLength; i++)
			{
				double val = weights(i);
				if (val != 0)
				{
					int ind = i+1;
					fh << ind << ":" << val << " ";
				}
			}
			fh << endl;

			fh.close();
		}
		else
		{
			cerr << "[Error] cannot open svmrank model file for writing weights!!" << endl;
			exit(1);
		}
	}

	int SVMRankModel::mergeRankingFiles(string fileNameBase, int numProcesses, int totalMasterQID)
	{
		int currentQID = totalMasterQID;

		string FEATURES_FILE = Global::settings->updateRankIDHelper(Global::settings->paths->OUTPUT_TEMP_DIR, fileNameBase, 0);
		cout << "Merging to main feature file: " << FEATURES_FILE << endl;

		ofstream ofh;
		ofh.open(FEATURES_FILE.c_str(), std::ios_base::app);

		if (ofh.is_open())
		{
			//ofh << endl; // add extra line
			for (int i = 1; i < numProcesses; i++)
			{
				// open file from process i
				FEATURES_FILE = Global::settings->updateRankIDHelper(Global::settings->paths->OUTPUT_TEMP_DIR, fileNameBase, i);

				ifstream fh;
				fh.open(FEATURES_FILE.c_str());

				string line;
				if (fh.is_open())
				{
					// loop over lines in file from process i
					// and append to the master file
					int prevSlaveQID = -1;
					while (fh.good())
					{
						// get line and split into parts
						getline(fh, line);
						if (line.empty())
							continue;

						stringstream ssLine(line);

						string sRanking;
						string sQID;
						string sFeatures;

						getline(ssLine, sRanking, ' ');
						getline(ssLine, sQID, ' ');
						getline(ssLine, sFeatures, ';'); // grab all the way to the end of the line

						stringstream ssQID(sQID);

						string sQIDToken;
						string sQIDValue;

						getline(ssQID, sQIDToken, ':');
						getline(ssQID, sQIDValue, ' ');

						int currentSlaveQID = atoi(sQIDValue.c_str());

						// adjust qid accordingly
						if (prevSlaveQID != currentSlaveQID)
						{
							currentQID++;
							prevSlaveQID = currentSlaveQID;
						}

						// append to master file
						ofh << sRanking << " " << "qid:" << currentQID << " " << sFeatures << endl;
					}

					fh.close();

					// delete the slave feature file
					ostringstream ossRemoveRankingFeatureCmd;
					ossRemoveRankingFeatureCmd << Global::settings->cmds->SYSTEM_RM_CMD 
						<< " " << FEATURES_FILE;
					MyFileSystem::Executable::execute(ossRemoveRankingFeatureCmd.str());
				}
				else
				{
					cerr << "[Error] master process could not open ranking file from a process: " << i << endl;
				}
			}

			ofh.close();
		}
		else
		{
			cerr << "[Error] master process could not open ranking file!" << endl;
		}

		return currentQID;
	}

	/**************** Online Rank Model ****************/

	OnlineRankModel::OnlineRankModel()
	{
		this->initialized = false;
	}

	OnlineRankModel::OnlineRankModel(string fileName)
	{
		load(fileName);
	}

	double OnlineRankModel::rank(RankFeatures features)
	{
		if (!initialized)
		{
			cout << "Initializing online rank weights..." << endl;
			initialize(features.data.size());
		}

		return vectorDot(getAvgWeights(), features.data);
	}

	RankerType OnlineRankModel::rankerType()
	{
		return ONLINE_RANK;
	}

	void OnlineRankModel::load(string fileName)
	{
		parseModelFile(fileName, this->latestWeights, this->cumSumWeights, this->numSum);
		this->initialized = true;
	}

	void OnlineRankModel::save(string fileName)
	{
		writeModelFile(fileName, this->latestWeights, this->cumSumWeights, this->numSum);
	}

	VectorXd OnlineRankModel::getLatestWeights()
	{
		if (!initialized)
		{
			cerr << "[Error] online ranker not initialized for getting latest weights" << endl;
			exit(1);
		}

		return this->latestWeights;
	}

	VectorXd OnlineRankModel::getAvgWeights()
	{
		if (!initialized)
		{
			cerr << "[Error] online ranker not initialized for getting avg weights" << endl;
			exit(1);
		}

		return (1.0/this->numSum)*this->cumSumWeights;
	}

	void OnlineRankModel::performOnlineUpdate(double delta, VectorXd featureDiff)
	{
		if (!initialized)
		{
			cout << "Initializing online rank weights..." << endl;
			initialize(featureDiff.size());
		}

		cout << "Performing online update..." << endl;

		double tauNumerator = sqrt(delta) - vectorDot(getLatestWeights(), featureDiff);
		double tauDenominator = pow(featureDiff.norm(), 2);

		if (tauDenominator == 0 && tauNumerator == 0)
		{
			cerr << "[Warning] tau indeterminant form error" << endl;
		}
		else if (tauDenominator == 0 && tauNumerator != 0)
		{
			cerr << "[Warning] tau division by zero error" << endl;
		}
		else
		{
			double tau = tauNumerator/tauDenominator;
			VectorXd newWeights = getLatestWeights() + tau * featureDiff;

			// perform update
			this->latestWeights = newWeights;
			this->cumSumWeights += newWeights;
			this->numSum += 1;
		}
	}

	void OnlineRankModel::initialize(int dim)
	{
		this->latestWeights = VectorXd::Zero(dim);
		this->cumSumWeights = VectorXd::Zero(dim);
		this->numSum = 1;
		this->initialized = true;
	}

	void OnlineRankModel::parseModelFile(string fileName, VectorXd& latestWeights, VectorXd& cumSumWeights, int& numSum)
	{
		string line;
		vector<int> indices;
		vector<double> values;
		vector<int> indices2;
		vector<double> values2;

		ifstream fh(fileName.c_str());
		if (fh.is_open())
		{
			int lineNum = 0;
			while (fh.good())
			{
				lineNum++;
				getline(fh, line);

				if (lineNum == 1)
				{
					numSum = atoi(line.c_str());
				}
				else if (lineNum == 2)
				{
					istringstream iss(line);

					string token;
					while (getline(iss, token, ' '))
					{
						if (token.find(':') == std::string::npos)
							continue;

						istringstream isstoken(token);
						string sIndex;
						string sValue;
						getline(isstoken, sIndex, ':');
						getline(isstoken, sValue, ':');

						int index = atoi(sIndex.c_str());
						double value = atof(sValue.c_str());

						indices.push_back(index);
						values.push_back(value);
					}
				}
				else if (lineNum == 3)
				{
					istringstream iss(line);

					string token;
					while (getline(iss, token, ' '))
					{
						if (token.find(':') == std::string::npos)
							continue;

						istringstream isstoken(token);
						string sIndex;
						string sValue;
						getline(isstoken, sIndex, ':');
						getline(isstoken, sValue, ':');

						int index = atoi(sIndex.c_str());
						double value = atof(sValue.c_str());

						indices2.push_back(index);
						values2.push_back(value);
					}
				}
			}
			fh.close();
		}
		else
		{
			cerr << "[Error] cannot open model file for reading weights!!" << endl;
			exit(1);
		}

		cumSumWeights = VectorXd::Zero(indices.back());
		int valuesSize = values.size();
		if (valuesSize == 0)
		{
			cerr << "[Error] assigning empty cumsumweights from '" + fileName + "'!" << endl;
			exit(1);
		}
		for (int i = 0; i < valuesSize; i++)
		{
			int ind = indices[i]-1;
			cumSumWeights(ind) = values[i];
		}

		latestWeights = VectorXd::Zero(indices.back());
		int valuesSize2 = values2.size();
		if (valuesSize2 == 0)
		{
			cerr << "[Error] assigning empty latestweights from '" + fileName + "'!" << endl;
			exit(1);
		}
		for (int i = 0; i < valuesSize2; i++)
		{
			int ind = indices2[i]-1;
			latestWeights(ind) = values2[i];
		}
	}

	void OnlineRankModel::writeModelFile(string fileName, const VectorXd& latestWeights, const VectorXd& cumSumWeights, int numSum)
	{
		ofstream fh(fileName.c_str());
		if (fh.is_open())
		{
			// write num to file
			fh << numSum << endl;

			// write cumulative sum weights to file
			const int cumSumWeightsLength = cumSumWeights.size();
			for (int i = 0; i < cumSumWeightsLength; i++)
			{
				double val = cumSumWeights(i);
				if (val != 0)
				{
					int ind = i+1;
					fh << ind << ":" << val << " ";
				}
			}
			fh << endl;

			// write latest weights to file
			const int latestWeightsLength = latestWeights.size();
			for (int i = 0; i < latestWeightsLength; i++)
			{
				double val = latestWeights(i);
				if (val != 0)
				{
					int ind = i+1;
					fh << ind << ":" << val << " ";
				}
			}
			fh << endl;

			fh.close();
		}
		else
		{
			cerr << "[Error] cannot open online model file for writing weights!!" << endl;
			exit(1);
		}
	}
}