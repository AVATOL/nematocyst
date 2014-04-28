#ifndef SUCCESSORFUNCTION_HPP
#define SUCCESSORFUNCTION_HPP

#include "DataStructures.hpp"
#include "MyGraphAlgorithms.hpp"

namespace HCSearch
{
	/**************** Search Space Function Abstract Definitions ****************/

	/*!
	 * @brief Successor function interface
	 */
	class ISuccessorFunction
	{
	public:
		virtual ~ISuccessorFunction() {}
		
		/*!
		 * @brief Generate successors.
		 */
		virtual vector< ImgCandidate > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred, int timeStep, int timeBound)=0;
	};

	/**************** Successor Functions ****************/

	/*!
	 * @brief Deterministic flipbit successor function.
	 *
	 * For each node, flip its label to all possible classes.
	 */
	class FlipbitSuccessor : public ISuccessorFunction
	{
	protected:
		static const double TOP_CONFIDENCES_PROPORTION;
		static const int NUM_TOP_LABELS_KEEP;
		static const double BINARY_CONFIDENCE_THRESHOLD;

	public:
		FlipbitSuccessor();
		~FlipbitSuccessor();
		
		virtual vector< ImgCandidate > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred, int timeStep, int timeBound);
	};

	/*!
	 * @brief Deterministic flipbit successor function using neighbor labels.
	 *
	 * For each node, flip its label to a label of a neighboring node.
	 */
	class FlipbitNeighborSuccessor : public FlipbitSuccessor
	{
	public:
		FlipbitNeighborSuccessor();
		~FlipbitNeighborSuccessor();
		
		virtual vector< ImgCandidate > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred, int timeStep, int timeBound);
	};

	/*!
	 * @brief Deterministic flipbit successor function using top K confidences and neighbor labels.
	 *
	 * For each node, flip its label to a label of a confident or neighboring node.
	 */
	class FlipbitConfidencesNeighborSuccessor : public FlipbitSuccessor
	{
	public:
		FlipbitConfidencesNeighborSuccessor();
		~FlipbitConfidencesNeighborSuccessor();
		
		virtual vector< ImgCandidate > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred, int timeStep, int timeBound);
	};

	/*!
	 * @brief Stochastic successor function.
	 * 
	 * Stochastically cut edges to form subgraphs. 
	 * For each subgraph, flip its label to all possible classes.
	 */
	class StochasticSuccessor : public ISuccessorFunction
	{
	protected:
		static const double TOP_CONFIDENCES_PROPORTION;
		static const double DEFAULT_T_PARM;
		double cutParam; //!< temperature parameter
		bool cutEdgesIndependently; //!< cut independently if true, cut by state otherwise

	public:
		StochasticSuccessor();
		StochasticSuccessor(bool cutEdgesIndependently, double cutParam);
		~StochasticSuccessor();

		virtual vector< ImgCandidate > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred, int timeStep, int timeBound);

	protected:
		virtual MyGraphAlgorithms::SubgraphSet* cutEdges(ImgFeatures& X, ImgLabeling& YPred, double threshold, double T);
		virtual vector< ImgCandidate > createCandidates(ImgLabeling& YPred, MyGraphAlgorithms::SubgraphSet* subgraphs);
		virtual void getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);

		void getAllLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);
		void getNeighborLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);
		void getConfidencesNeighborLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);

		static double computeKL(const VectorXd& p, const VectorXd& q);
	};

	/*!
	 * @brief Stochastic successor function using neighbor labels.
	 * 
	 * Stochastically cut edges to form subgraphs. 
	 * For each subgraph, flip its label to a label of a neighboring node.
	 */
	class StochasticNeighborSuccessor : public StochasticSuccessor
	{
	public:
		StochasticNeighborSuccessor();
		StochasticNeighborSuccessor(bool cutEdgesIndependently, double cutParam);
		~StochasticNeighborSuccessor();

	protected:
		virtual void getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);
	};

	/*!
	 * @brief Stochastic successor function using confident labels and neighbor labels.
	 * 
	 * Stochastically cut edges to form subgraphs. 
	 * For each subgraph, flip its label to a label of a confident or neighboring node.
	 */
	class StochasticConfidencesNeighborSuccessor : public StochasticSuccessor
	{
	public:
		StochasticConfidencesNeighborSuccessor();
		StochasticConfidencesNeighborSuccessor(bool cutEdgesIndependently, double cutParam);
		~StochasticConfidencesNeighborSuccessor();

	protected:
		virtual void getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);
	};

	/*!
	 * @brief Cut schedule successor function.
	 * 
	 * Schedule to find the best cut for forming subgraphs. 
	 * For each subgraph, flip its label to all possible classes.
	 */
	class CutScheduleSuccessor : public StochasticSuccessor
	{
	protected:
		static const int NUM_GOOD_SUBGRAPHS_THRESHOLD;
		static const double FINAL_THRESHOLD;
		static const double THRESHOLD_INCREMENT;

	public:
		CutScheduleSuccessor();
		CutScheduleSuccessor(double cutParam);
		~CutScheduleSuccessor();

		virtual vector< ImgCandidate > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred, int timeStep, int timeBound);

	protected:
		virtual MyGraphAlgorithms::SubgraphSet* cutEdges(ImgFeatures& X, ImgLabeling& YPred, double threshold, double T);
		virtual void getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);
	};

	/*!
	 * @brief Cut schedule successor function using neighbor labels.
	 * 
	 * Schedule to find the best cut for forming subgraphs. 
	 * For each subgraph, flip its label to a label of a neighboring node.
	 */
	class CutScheduleNeighborSuccessor : public CutScheduleSuccessor
	{
	public:
		CutScheduleNeighborSuccessor();
		CutScheduleNeighborSuccessor(double cutParam);
		~CutScheduleNeighborSuccessor();

	protected:
		virtual void getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);
	};

	/*!
	 * @brief Cut schedule successor function using confident labels and neighbor labels.
	 * 
	 * Schedule to find the best cut for forming subgraphs. 
	 * For each subgraph, flip its label to a label of a confident or neighboring node.
	 */
	class CutScheduleConfidencesNeighborSuccessor : public CutScheduleSuccessor
	{
	public:
		CutScheduleConfidencesNeighborSuccessor();
		CutScheduleConfidencesNeighborSuccessor(double cutParam);
		~CutScheduleConfidencesNeighborSuccessor();

	protected:
		virtual void getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);
	};

	/*!
	 * @brief Stochastic schedule successor function.
	 * 
	 * Stochastically cut edges to form subgraphs. 
	 * For each subgraph, flip its label to all possible classes.
	 */
	class StochasticScheduleSuccessor : public ISuccessorFunction
	{
	protected:
		static const double TOP_CONFIDENCES_PROPORTION;
		static const double DEFAULT_T_PARAM;
		static const double DEFAULT_NODE_CLAMP_THRESHOLD;
		static const double DEFAULT_EDGE_CLAMP_POSITIVE_THRESHOLD;
		static const double DEFAULT_EDGE_CLAMP_NEGATIVE_THRESHOLD;

		double cutParam; //!< temperature parameter
		bool cutEdgesIndependently; //!< cut independently if true, cut by state otherwise

		bool clampNodes;
		bool clampEdges;

		double nodeClampThreshold;
		double edgeClampPositiveThreshold;
		double edgeClampNegativeThreshold;

	public:
		StochasticScheduleSuccessor();
		StochasticScheduleSuccessor(bool cutEdgesIndependently, double cutParam);
		StochasticScheduleSuccessor(bool cutEdgesIndependently, double cutParam, 
			bool clampNodes, bool clampEdges, double nodeClampThreshold, double edgeClampPositiveThreshold, double edgeClampNegativeThreshold);
		~StochasticScheduleSuccessor();

		virtual vector< ImgCandidate > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred, int timeStep, int timeBound);

	protected:
		virtual MyGraphAlgorithms::SubgraphSet* cutEdges(ImgFeatures& X, ImgLabeling& YPred, double threshold, double T, int timeStep, int timeBound);
		virtual vector< ImgCandidate > createCandidates(ImgLabeling& YPred, MyGraphAlgorithms::SubgraphSet* subgraphs);
		virtual void getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);

		void getAllLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);
		void getNeighborLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);
		void getConfidencesNeighborLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);

		static double computeKL(const VectorXd& p, const VectorXd& q);
	};

	/*!
	 * @brief Stochastic schedule successor function using neighbor labels.
	 * 
	 * Stochastically cut edges to form subgraphs. 
	 * For each subgraph, flip its label to a label of a neighboring node.
	 */
	class StochasticScheduleNeighborSuccessor : public StochasticScheduleSuccessor
	{
	public:
		StochasticScheduleNeighborSuccessor();
		StochasticScheduleNeighborSuccessor(bool cutEdgesIndependently, double cutParam);
		StochasticScheduleNeighborSuccessor(bool cutEdgesIndependently, double cutParam, 
			bool clampNodes, bool clampEdges, double nodeClampThreshold, double edgeClampPositiveThreshold, double edgeClampNegativeThreshold);
		~StochasticScheduleNeighborSuccessor();

	protected:
		virtual void getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);
	};

	/*!
	 * @brief Stochastic schedule successor function using confident labels and neighbor labels.
	 * 
	 * Stochastically cut edges to form subgraphs. 
	 * For each subgraph, flip its label to a label of a confident or neighboring node.
	 */
	class StochasticScheduleConfidencesNeighborSuccessor : public StochasticScheduleSuccessor
	{
	public:
		StochasticScheduleConfidencesNeighborSuccessor();
		StochasticScheduleConfidencesNeighborSuccessor(bool cutEdgesIndependently, double cutParam);
		StochasticScheduleConfidencesNeighborSuccessor(bool cutEdgesIndependently, double cutParam, 
			bool clampNodes, bool clampEdges, double nodeClampThreshold, double edgeClampPositiveThreshold, double edgeClampNegativeThreshold);
		~StochasticScheduleConfidencesNeighborSuccessor();

	protected:
		virtual void getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);
	};
}

#endif