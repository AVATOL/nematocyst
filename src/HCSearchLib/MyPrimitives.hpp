#ifndef MYPRIMITIVES_HPP
#define MYPRIMITIVES_HPP

#include <map>
#include <set>

using namespace std;

//////////////////////////////////////////
/// My Custom Primitive Data Structures
//////////////////////////////////////////

namespace MyPrimitives
{
	template <class T, class U> struct Pair;
	template <class T, class U, class V> struct Triple;
	template <class T> class Bimap;

	/*!
	 * Pair stores a basic 2-tuple
	 */
	template <class T, class U=T>
	struct Pair
	{
		T first;
		U second;

		Pair(T first, U second)
		{
			this->first = first;
			this->second = second;
		}

		Pair() {}
		~Pair() {}
	};

	/*!
	 * Triple stores a basic 3-tuple
	 */
	template <class T, class U=T, class V=T>
	struct Triple
	{
		T first;
		U second;
		V third;

		Triple(T first, U second, V third)
		{
			this->first = first;
			this->second = second;
			this->third = third;
		}

		Triple() {}
		~Triple() {}
	};

	/*!
	 * Bimap stores a bijective map (one-to-one) with lookups in both directions
	 */
	template <class T>
	class Bimap
	{
		map<T, T> forward;
		map<T, T> backward;

		set<T> forwardKeys;
		set<T> backwardKeys;

	public:
		Bimap()
		{
			this->forward = map<T, T>();
			this->backward = map<T, T>();
			this->forwardKeys = set<T>();
			this->backwardKeys = set<T>();
		}
		~Bimap() {};
		
		/*!
		 * Insert element X and Y as a bijective pair
		 * X must not exist in the forward set
		 * and Y must not exist in the backward set
		 */
		void insert(T X, T Y);

		/*!
		 * Removes element X from the forward set 
		 * and its corresponding Y from the backward set
		 */
		void remove(T X);

		/*!
		 * Removes element Y from the backward set 
		 * and its corresponding X from the forward set
		 */
		void iremove(T Y);

		/*!
		 * Checks if element X exists in the forward set
		 */
		bool exists(T X);

		/*!
		 * Checks if element Y exists in the backward set
		 */
		bool iexists(T Y);

		/*!
		 * Looks up the corresponding element of element X in the forward set
		 */
		T lookup(T X);

		/*!
		 * Looks up the corresponding element of element Y in the backward set
		 */
		T ilookup(T Y);

		/*!
		 * Returns the forward set
		 */
		set<T> keyset();

		/*!
		 * Returns the backward set
		 */
		set<T> ikeyset();

		/*!
		 * Get the size of the bijective map
		 */
		int size();

		/*!
		 * Check if the bijective map is empty
		 */
		bool empty();

		/*!
		 * Reset the bijective map to zero elements
		 */
		void clear();
	};

	template <class T>
	void Bimap<T>::insert(T X, T Y)
	{
		if (exists(X) || iexists(Y))
			throw 1;

		this->forward[X] = Y;
		this->backward[Y] = X;

		this->forwardKeys.insert(X);
		this->backwardKeys.insert(Y);
	}

	template <class T>
	void Bimap<T>::remove(T X)
	{
		if (!exists(X))
			throw 1;

		this->backward.erase(forward[X]);
		this->backwardKeys.erase(forward[X]);
		this->forward.erase(X);
		this->forwardKeys.erase(X);
	}

	template <class T>
	void Bimap<T>::iremove(T Y)
	{
		if (!iexists(Y))
			throw 1;

		this->forward.erase(backward[Y]);
		this->forwardKeys.erase(backward[Y]);
		this->backward.erase(Y);
		this->backwardKeys.erase(Y);
	}

	template <class T>
	bool Bimap<T>::exists(T X)
	{
		return this->forward.count(X) == 1;
	}

	template <class T>
	bool Bimap<T>::iexists(T Y)
	{
		return this->backward.count(Y) == 1;
	}

	template <class T>
	T Bimap<T>::lookup(T X)
	{
		if (!exists(X))
			throw 1;

		return this->forward[X];
	}

	template <class T>
	T Bimap<T>::ilookup(T Y)
	{
		if (!iexists(Y))
			throw 1;

		return this->backward[Y];
	}

	template <class T>
	set<T> Bimap<T>::keyset()
	{
		return this->forwardKeys;
	}

	template <class T>
	set<T> Bimap<T>::ikeyset()
	{
		return this->backwardKeys;
	}

	template <class T>
	int Bimap<T>::size()
	{
		return this->forward.size();
	}

	template <class T>
	bool Bimap<T>::empty()
	{
		return size() == 0;
	}

	template <class T>
	void Bimap<T>::clear()
	{
		this->forward.clear();
		this->backward.clear();
		this->fowardKeys.clear();
		this->backwardKeys.clear();
	}
}

#endif