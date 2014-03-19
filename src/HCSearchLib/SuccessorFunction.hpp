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
	protected:
		int maxNumSuccessorCandidates;

	public:
		virtual ~ISuccessorFunction() {}
		
		/*!
		 * @brief Generate successors.
		 */
		virtual vector< ImgLabeling > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred)=0;
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
		FlipbitSuccessor(int maxNumSuccessorCandidates);
		~FlipbitSuccessor();
		
		virtual vector< ImgLabeling > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred);
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
		FlipbitNeighborSuccessor(int maxNumSuccessorCandidates);
		~FlipbitNeighborSuccessor();
		
		virtual vector< ImgLabeling > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred);
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
		FlipbitConfidencesNeighborSuccessor(int maxNumSuccessorCandidates);
		~FlipbitConfidencesNeighborSuccessor();
		
		virtual vector< ImgLabeling > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred);
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
		StochasticSuccessor(bool cutEdgesIndependently, double cutParam, int maxNumSuccessorCandidates);
		~StochasticSuccessor();

		virtual vector< ImgLabeling > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred);

	protected:
		virtual MyGraphAlgorithms::SubgraphSet* cutEdges(ImgFeatures& X, ImgLabeling& YPred, double threshold, double T);
		virtual vector< ImgLabeling > createCandidates(ImgLabeling& YPred, MyGraphAlgorithms::SubgraphSet* subgraphs);
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
		StochasticNeighborSuccessor(bool cutEdgesIndependently, double cutParam, int maxNumSuccessorCandidates);
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
		StochasticConfidencesNeighborSuccessor(bool cutEdgesIndependently, double cutParam, int maxNumSuccessorCandidates);
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
		CutScheduleSuccessor(double cutParam, int maxNumSuccessorCandidates);
		~CutScheduleSuccessor();

		virtual vector< ImgLabeling > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred);

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
		CutScheduleNeighborSuccessor(double cutParam, int maxNumSuccessorCandidates);
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
		CutScheduleConfidencesNeighborSuccessor(double cutParam, int maxNumSuccessorCandidates);
		~CutScheduleConfidencesNeighborSuccessor();

	protected:
		virtual void getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);
	};
}

#endif