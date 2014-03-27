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

	const string SearchTypeStrings[] = {"ll", "hl", "lc", "hc", "learnh", "learnc", "learncoracle", "rl", "rc", "learncrandom", "learndecomposed", "learnp", "discoverpairwise"};
	const string DatasetTypeStrings[] = {"test", "train", "validation"};

	/**************** Priority Queues ****************/

	bool CompareByConfidence::operator() (MyPrimitives::Pair<int, double>& lhs, MyPrimitives::Pair<int, double>& rhs) const
	{
		return lhs.second < rhs.second;
	}

	/**************** Features and Labelings ****************/

	ImgFeatures::ImgFeatures()
	{
		this->filename = "";
		this->segmentsAvailable = false;
		this->nodeLocationsAvailable = false;
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

	double ImgFeatures::getNodeLocationX(int node)
	{
		if (!nodeLocationsAvailable)
		{
			LOG(ERROR) << "node location not available for getting x.";
			abort();
		}

		return this->nodeLocations(node, 0);
	}

	double ImgFeatures::getNodeLocationY(int node)
	{
		if (!nodeLocationsAvailable)
		{
			LOG(ERROR) << "node location not available for getting y.";
			abort();
		}

		return this->nodeLocations(node, 1);
	}

	ImgLabeling::ImgLabeling()
	{
		this->confidencesAvailable = false;
		this->stochasticCutsAvailable = false;
		this->nodeWeightsAvailable = false;
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

	set<int> ImgLabeling::getTopConfidentLabels(int node, int K)
	{
		// check if confidences are available
		if (!this->confidencesAvailable)
		{
			LOG(ERROR) << "confidences are not available to get top K confident labels.";
			abort();
		}

		// bad cases
		const int numLabels = this->confidences.cols();
		if (K > numLabels)
		{
			return HCSearch::Global::settings->CLASSES.getLabels();
		}
		else if (K == 0)
		{
			return set<int>();
		}
		else if (K < 0)
		{
			LOG(ERROR) << "K cannot be negative!";
			HCSearch::abort();
		}

		set<int> labels;

		// get top K confident labels
		LabelConfidencePQ sortedByConfidence;
		for (int i = 0; i < numLabels; i++)
		{
			int label = HCSearch::Global::settings->CLASSES.getClassLabel(i);
			double confidence = this->confidences(node, i);
			sortedByConfidence.push(MyPrimitives::Pair<int, double>(label, confidence));
		}
		for (int i = 0; i < K; i++)
		{
			MyPrimitives::Pair<int, double> p = sortedByConfidence.top();
			sortedByConfidence.pop();
			labels.insert(p.first);
		}

		return labels;
	}

	vector<int> ImgLabeling::getLabelsByConfidence(int node)
	{
		// check if confidences are available
		if (!this->confidencesAvailable)
		{
			LOG(ERROR) << "confidences are not available to get sorted confident labels.";
			abort();
		}

		const int numLabels = this->confidences.cols();
		vector<int> labels;

		// get top K confident labels
		LabelConfidencePQ sortedByConfidence;
		for (int i = 0; i < numLabels; i++)
		{
			int label = HCSearch::Global::settings->CLASSES.getClassLabel(i);
			double confidence = this->confidences(node, i);
			sortedByConfidence.push(MyPrimitives::Pair<int, double>(label, confidence));
		}
		for (int i = 0; i < numLabels; i++)
		{
			MyPrimitives::Pair<int, double> p = sortedByConfidence.top();
			sortedByConfidence.pop();
			labels.push_back(p.first);
		}

		return labels;
	}

	/**************** Classify/Rank Features ****************/

	GenericFeatures::GenericFeatures()
	{
	}

	GenericFeatures::GenericFeatures(VectorXd features)
	{
		this->data = features;
	}

	GenericFeatures::~GenericFeatures()
	{
	}

	/**************** SVM Classifier Model ****************/

	SVMClassifierModel::SVMClassifierModel()
	{
		this->isClassifyReady = false;
		this->isTraining = false;
	}

	SVMClassifierModel::SVMClassifierModel(string fileName)
	{
		load(fileName);
	}

	int SVMClassifierModel::classify(ClassifierFeatures features)
	{
		vector<ClassifierFeatures> featureWrapper;
		featureWrapper.push_back(features);
		vector<int> results = classify(featureWrapper);
		return results[0];
	}

	vector<int> SVMClassifierModel::classify(vector<ClassifierFeatures> features)
	{
		if (!this->isClassifyReady)
		{
			LOG(ERROR) << "cannot classify from uninitialized svm model object";
			abort();
		}

		// write to file
		writeFeaturesToFile(features, Global::settings->paths->OUTPUT_CLASSIFIER_FEATURES_FILE);

		// run SVM program
		stringstream ssPredictCmd;
		ssPredictCmd << Global::settings->cmds->LIBSVM_PREDICT_CMD << " " 
			<< Global::settings->paths->OUTPUT_CLASSIFIER_FEATURES_FILE << " " + this->modelFileName 
			<< " " << Global::settings->paths->OUTPUT_CLASSIFIER_PREDICT_FILE;

		int retcode = MyFileSystem::Executable::executeRetries(ssPredictCmd.str());
		if (retcode != 0)
		{
			LOG(ERROR) << "SVM classifier failed!";
			abort();
		}

		// read back in results
		vector<int> labels;
		MatrixXd confidences;
		readLabelsFromFile(labels, confidences, features.size(), Global::settings->paths->OUTPUT_CLASSIFIER_PREDICT_FILE);

		return labels;
	}

	ClassifierType SVMClassifierModel::classifierType()
	{
		return SVM_CLASSIFIER;
	}

	void SVMClassifierModel::load(string fileName)
	{
		this->modelFileName = fileName;
		this->isClassifyReady = true;
	}

	void SVMClassifierModel::save(string fileName)
	{
		if (!this->isClassifyReady)
		{
			LOG(WARNING) << "cannot save svm model from uninitialized object";
			return;
		}

		MyFileSystem::FileSystem::copyFile(this->modelFileName, fileName);
	}

	void SVMClassifierModel::startTraining(string featuresFileName)
	{
		this->isTraining = true;
		this->trainingFileStream = new ofstream(featuresFileName.c_str());
		this->trainingFileName = featuresFileName;
	}

	void SVMClassifierModel::addTrainingExample(ClassifierFeatures& features, int label)
	{
		(*this->trainingFileStream) << vector2svm(features, label) << endl;
	}

	void SVMClassifierModel::finishTraining(string modelFileName)
	{
		// close ranking file
		this->trainingFileStream->close();
		delete this->trainingFileStream;

#ifdef USE_MPI
		string STARTMSG;
		string ENDMSG;
		string featuresFileBase;
		STARTMSG = "MERGEPSTART";
		ENDMSG = "MERGEPEND";
		featuresFileBase = Global::settings->paths->OUTPUT_PRUNE_FEATURES_FILE_BASE;

		MPI::Synchronize::masterWait(STARTMSG);

		// merge step
		if (Global::settings->RANK == 0)
		{
			mergeFeatureFiles(featuresFileBase, Global::settings->NUM_PROCESSES);
		}
#endif

		if (Global::settings->RANK == 0)
		{
			clock_t tic = clock();

			// compute C
			double C = 1.0;

			// call SVM-Rank
			stringstream ssLearn;
			ssLearn << Global::settings->cmds->LIBSVM_TRAIN_CMD << " -t 0 -c " << C << " " 
				<< this->trainingFileName << " " << modelFileName;
			MyFileSystem::Executable::executeRetries(ssLearn.str());

			clock_t toc = clock();
			LOG() << "total SVM classifier training time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl;
		}

#ifdef USE_MPI
		MPI::Synchronize::slavesWait(ENDMSG);
#endif

		// no longer learning
		this->isTraining = false;

		// load weights into model and initialize
		if (Global::settings->RANK == 0)
		{
			load(modelFileName);
		}

		LOG() << endl;
	}

	void SVMClassifierModel::cancelTraining()
	{
		// close training file
		this->trainingFileStream->close();
		delete this->trainingFileStream;

		// no longer learning
		this->isTraining = false;
	}

	void SVMClassifierModel::writeFeaturesToFile(vector<ClassifierFeatures> features, string fileName)
	{
		ofstream fh(fileName.c_str());
		if (fh.is_open())
		{
			for (vector<ClassifierFeatures>::iterator it = features.begin(); it != features.end(); ++it)
			{
				ClassifierFeatures featureVector = *it;
				const int dummyLabel = 1; // dummy for prediction
				fh << vector2svm(featureVector, dummyLabel) << endl;
			}
			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open file for writing LIBSVM classifier features!";
			abort();
		}
	}

	void SVMClassifierModel::readLabelsFromFile(vector<int>& labels, MatrixXd& confidences, int numLabels, string fileName)
	{
		vector<int> labelOrderFound;
		//TODO: handle confidences if available
		bool confidencesAvailable = false;

		int lineIndex = 0;
		string line;
		ifstream fh(fileName.c_str());
		if (fh.is_open())
		{
			int numClassesFound = 0;
			while (fh.good() && lineIndex < numLabels+1)
			{
				getline(fh, line);
				if (!line.empty())
				{
					// parse line to get label and confidences
					stringstream ss(line);
					string token;
					int columnIndex = 0;
					while (getline(ss, token, ' '))
					{
						if (columnIndex == 0)
						{
							int label = atoi(token.c_str());
							labels.push_back(label);
						}
						columnIndex++;
					}
				}

				lineIndex++;
			}
			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open file for reading LIBSVM labels/confidences!";
			abort();
		}
	}

	string SVMClassifierModel::vector2svm(ClassifierFeatures features, int label)
	{
		stringstream ss;
		ss << label;
		const int numFeatures = features.data.size();
		for (int i = 0; i < numFeatures; i++)
		{
			double feat = features.data[i];
			if (feat != 0)
				ss << " " << i << ":" << feat;
		}

		return ss.str();
	}
	
	void SVMClassifierModel::mergeFeatureFiles(string fileNameBase, int numProcesses)
	{
		string FEATURES_FILE = Global::settings->updateRankIDHelper(Global::settings->paths->OUTPUT_TEMP_DIR, fileNameBase, 0);
		LOG() << "Merging to main feature file: " << FEATURES_FILE << endl;

		ofstream ofh;
		ofh.open(FEATURES_FILE.c_str(), std::ios_base::app);

		if (ofh.is_open())
		{
			//ofh << endl; // add extra line
			for (int i = 1; i < numProcesses; i++)
			{
				// open file from process i
				FEATURES_FILE = Global::settings->updateRankIDHelper(Global::settings->paths->OUTPUT_TEMP_DIR, fileNameBase, i);

				if (!MyFileSystem::FileSystem::checkFileExists(FEATURES_FILE))
				{
					continue;
				}

				ifstream fh;
				fh.open(FEATURES_FILE.c_str());

				string line;
				if (fh.is_open())
				{
					// loop over lines in file from process i
					// and append to the master file
					while (fh.good())
					{
						// get line
						getline(fh, line);
						if (line.empty())
							continue;

						// append to master file
						ofh << line << endl;
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
					LOG(ERROR) << "master process could not open training file from a process: " << i;
				}
			}

			ofh.close();
		}
		else
		{
			LOG(ERROR) << "master process could not open training file!";
		}
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
			LOG(ERROR) << "svm ranker not initialized for ranking";
			abort();
		}

		return vectorDot(getWeights(), features.data);
	}

	RankerType SVMRankModel::rankerType()
	{
		return SVM_RANK;
	}

	void SVMRankModel::load(string fileName)
	{
		if (!MyFileSystem::FileSystem::checkFileExists(fileName))
		{
			LOG(WARNING) << "SVM model file does not exist for loading! Ignoring load function...";
			return;
		}

		this->modelFileName = fileName;
		this->weights = parseModelFile(fileName);
		this->initialized = true;
	}

	void SVMRankModel::save(string fileName)
	{
		MyFileSystem::FileSystem::copyFile(this->modelFileName, fileName);
		//writeModelFile(fileName, this->weights);
	}
	
	VectorXd SVMRankModel::getWeights()
	{
		if (!this->initialized)
		{
			LOG(ERROR) << "svm ranker not initialized for getting weights";
			abort();
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
		int betterSetSize = betterSet.size();
		int worseSetSize = worseSet.size();

		LOG() << "Training with " << betterSetSize << " best examples and " << worseSetSize << " worst examples..." << endl;

		// Prune examples for efficiency vs. accuracy trade-off
		if (Global::settings->PRUNE_SVM_RANK_EXAMPLES)
		{
			// prune best set
			if (betterSetSize > Global::settings->PRUNE_SVM_RANK_MIN_EXAMPLES)
			{
				int newBetterSize = static_cast<int>(Global::settings->PRUNE_SVM_RANK_RATIO * betterSetSize);
				if (newBetterSize <= Global::settings->PRUNE_SVM_RANK_MIN_EXAMPLES)
					newBetterSize = Global::settings->PRUNE_SVM_RANK_MIN_EXAMPLES;
				if (newBetterSize > Global::settings->PRUNE_SVM_RANK_MAX_EXAMPLES)
					newBetterSize = Global::settings->PRUNE_SVM_RANK_MAX_EXAMPLES;

				random_shuffle(betterSet.begin(), betterSet.end());
				for (int i = 0; i < betterSetSize - newBetterSize; i++)
					betterSet.pop_back();

				betterSetSize = betterSet.size();
			}

			// prune worst set
			if (worseSetSize > Global::settings->PRUNE_SVM_RANK_MIN_EXAMPLES)
			{
				int newWorseSize = static_cast<int>(Global::settings->PRUNE_SVM_RANK_RATIO * worseSetSize);
				if (newWorseSize <= Global::settings->PRUNE_SVM_RANK_MIN_EXAMPLES)
					newWorseSize = Global::settings->PRUNE_SVM_RANK_MIN_EXAMPLES;
				if (newWorseSize > Global::settings->PRUNE_SVM_RANK_MAX_EXAMPLES)
					newWorseSize = Global::settings->PRUNE_SVM_RANK_MAX_EXAMPLES;

				random_shuffle(worseSet.begin(), worseSet.end());
				for (int i = 0; i < worseSetSize - newWorseSize; i++)
					worseSet.pop_back();

				worseSetSize = worseSet.size();
			}

			LOG() << "\tPruned to " << betterSetSize << " best examples and " << worseSetSize << " worst examples..." << endl;
		}

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
		if (searchType != LEARN_H && searchType != LEARN_C && searchType != LEARN_C_ORACLE_H && searchType != LEARN_C_RANDOM_H && searchType != LEARN_DECOMPOSED )
		{
			LOG(ERROR) << "invalid search type for training.";
			abort();
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
		else if (searchType == LEARN_C_RANDOM_H)
		{
			STARTMSG = "MERGECRHSTART";
			ENDMSG = "MERGECRHEND";
			featuresFileBase = Global::settings->paths->OUTPUT_COST_RANDOM_H_FEATURES_FILE_BASE;
		}
		else if (searchType == LEARN_DECOMPOSED)
		{
			STARTMSG = "MERGEDSTART";
			ENDMSG = "MERGEDEND";
			featuresFileBase = Global::settings->paths->OUTPUT_DECOMPOSED_LEARNING_FEATURES_FILE_BASE;
		}

		MPI::Synchronize::masterWait(STARTMSG);

		// merge step
		if (Global::settings->RANK == 0)
		{
			this->qid = mergeRankingFiles(featuresFileBase, Global::settings->NUM_PROCESSES, this->qid);
		}
#endif

		if (this->qid <= 1)
		{
			LOG(ERROR) << "no training data available for learning!";
		}
		else if (Global::settings->RANK == 0)
		{
			clock_t tic = clock();

			// compute C
			double C = 1.0 * (this->qid-1);

			// call SVM-Rank
			stringstream ssLearn;
			ssLearn << Global::settings->cmds->SVMRANK_LEARN_CMD << " -c " << C << " " 
				<< this->rankingFileName << " " << modelFileName;
			MyFileSystem::Executable::executeRetries(ssLearn.str());

			clock_t toc = clock();
			LOG() << "total SVM-Rank training time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl;
		}

#ifdef USE_MPI
		MPI::Synchronize::slavesWait(ENDMSG);
#endif

		// no longer learning
		this->learningMode = false;

		if (this->qid <= 1)
		{
			return;
		}

		// load weights into model and initialize
		if (Global::settings->RANK == 0)
		{
			load(modelFileName);
		}

		LOG() << endl;
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
			LOG(ERROR) << "cannot open model file for reading weights!!";
			abort();
		}

		int valuesSize = values.size();
		if (valuesSize == 0)
		{
			LOG(ERROR) << "found empty weights from '" + fileName + "'!";
			abort();
		}
		weights = VectorXd::Zero(indices.back());

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
			LOG(ERROR) << "cannot open svmrank model file for writing weights!!";
			abort();
		}
	}

	int SVMRankModel::mergeRankingFiles(string fileNameBase, int numProcesses, int totalMasterQID)
	{
		string FEATURES_FILE = Global::settings->updateRankIDHelper(Global::settings->paths->OUTPUT_TEMP_DIR, fileNameBase, 0);
		LOG() << "Merging to main feature file: " << FEATURES_FILE << endl;

		if (numProcesses == 1)
			return totalMasterQID;

		int currentQID = totalMasterQID-1;

		ofstream ofh;
		ofh.open(FEATURES_FILE.c_str(), std::ios_base::app);

		if (ofh.is_open())
		{
			//ofh << endl; // add extra line
			for (int i = 1; i < numProcesses; i++)
			{
				// open file from process i
				FEATURES_FILE = Global::settings->updateRankIDHelper(Global::settings->paths->OUTPUT_TEMP_DIR, fileNameBase, i);

				if (!MyFileSystem::FileSystem::checkFileExists(FEATURES_FILE))
				{
					continue;
				}

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
					LOG(ERROR) << "master process could not open ranking file from a process: " << i;
				}
			}

			ofh.close();
		}
		else
		{
			LOG(ERROR) << "master process could not open ranking file!";
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
			LOG() << "Initializing online rank weights..." << endl;
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
			LOG(ERROR) << "online ranker not initialized for getting latest weights";
			abort();
		}

		return this->latestWeights;
	}

	VectorXd OnlineRankModel::getAvgWeights()
	{
		if (!initialized)
		{
			LOG(ERROR) << "online ranker not initialized for getting avg weights";
			abort();
		}

		return (1.0/this->numSum)*this->cumSumWeights;
	}

	void OnlineRankModel::performOnlineUpdate(double delta, VectorXd featureDiff)
	{
		if (!initialized)
		{
			LOG() << "Initializing online rank weights..." << endl;
			initialize(featureDiff.size());
		}

		LOG() << "Performing online update..." << endl;

		double tauNumerator = sqrt(delta) - vectorDot(getLatestWeights(), featureDiff);
		double tauDenominator = pow(featureDiff.norm(), 2);

		if (tauDenominator == 0 && tauNumerator == 0)
		{
			LOG(WARNING) << "tau indeterminant form error";
		}
		else if (tauDenominator == 0 && tauNumerator != 0)
		{
			LOG(WARNING) << "tau division by zero error";
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

	void OnlineRankModel::performMerge(string modelFileBase, SearchType searchType)
	{
		mergeRankingFiles(this->latestWeights, this->cumSumWeights, this->numSum, modelFileBase, Global::settings->NUM_PROCESSES);
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
			LOG(ERROR) << "cannot open model file for reading weights!!";
			abort();
		}

		int valuesSize = values.size();
		if (valuesSize == 0)
		{
			LOG(ERROR) << "found empty cumsumweights from '" + fileName + "'!";
			abort();
		}
		cumSumWeights = VectorXd::Zero(indices.back());
		for (int i = 0; i < valuesSize; i++)
		{
			int ind = indices[i]-1;
			cumSumWeights(ind) = values[i];
		}

		int valuesSize2 = values2.size();
		if (valuesSize2 == 0)
		{
			LOG(ERROR) << "found empty latestweights from '" + fileName + "'!";
			abort();
		}
		latestWeights = VectorXd::Zero(indices.back());
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
			LOG(ERROR) << "cannot open online model file for writing weights!!";
			abort();
		}
	}

	void OnlineRankModel::mergeRankingFiles(VectorXd& masterLatestWeights, VectorXd& masterCumSumWeights, int& masterNumSum, string fileNameBase, int numProcesses)
	{
		LOG() << "Merging online model files...";

		vector< VectorXd > latestWeights;
		vector< VectorXd > weights;
		vector< int > sums;
		int largestDim = 0;

		for (int rank = 0; rank < numProcesses; rank++)
		{
			string WEIGHTS_FILE = Global::settings->updateRankIDHelper(Global::settings->paths->OUTPUT_TEMP_DIR, fileNameBase, rank);
			cout << WEIGHTS_FILE << endl;

			VectorXd lw;
			VectorXd w;
			int s;
			parseModelFile(WEIGHTS_FILE, lw, w, s);

			if (w.size() > largestDim)
				largestDim = w.size();

			latestWeights.push_back(lw);
			weights.push_back(w);
			sums.push_back(s);
		}

		masterLatestWeights = VectorXd::Zero(largestDim);
		masterCumSumWeights = VectorXd::Zero(largestDim);
		masterNumSum = 0;

		for (int rank = 0; rank < numProcesses; rank++)
		{
			masterLatestWeights += latestWeights[rank];
			masterCumSumWeights += weights[rank];
			masterNumSum += sums[rank];
		}

		masterLatestWeights /= numProcesses;

		LOG() << "...done merging online model files." << endl;
	}
}