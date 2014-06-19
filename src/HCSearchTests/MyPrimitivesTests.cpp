#include "stdafx.h"
#include "CppUnitTest.h"

#include "MyPrimitives.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace MyPrimitives;

namespace Testing
{		
	TEST_CLASS(MyPrimitivesTests)
	{
	public:
		TEST_METHOD(PairInMaps)
		{
			const double CONSTANT1 = 0.33;
			const double CONSTANT2 = 0.75;
			const double CONSTANT3 = 0.314;
			const double CONSTANT4 = 0.11;

			// construct
			map< Pair<int, int>, double> test;

			// insert element
			test[Pair<int, int>(0, 1)] = CONSTANT1;
			Assert::AreEqual(test[Pair<int, int>(0, 1)], CONSTANT1);

			// insert another element
			test[Pair<int, int>(1, 0)] = CONSTANT2;
			Assert::AreEqual(test[Pair<int, int>(1, 0)], CONSTANT2);

			// insert duplicate element
			test[Pair<int, int>(0, 1)] = CONSTANT3;
			Assert::AreEqual(test[Pair<int, int>(0, 1)], CONSTANT3);

			Assert::AreEqual(static_cast<int>(test.count(Pair<int, int>(0, 1))), 1);

			Assert::AreEqual(static_cast<int>(test.count(Pair<int, int>(1, 1))), 0);
		}

		TEST_METHOD(PairInSets)
		{
			// construct
			set< Pair<int, int> > test;

			// insert element
			test.insert(Pair<int, int>(0, 1));
			Assert::AreEqual(static_cast<int>(test.count(Pair<int, int>(0, 1))), 1);
			Assert::AreEqual(static_cast<int>(test.size()), 1);

			//// insert another element
			test.insert(Pair<int, int>(1, 0));
			Assert::AreEqual(static_cast<int>(test.count(Pair<int, int>(1, 0))), 1);
			Assert::AreEqual(static_cast<int>(test.size()), 2);

			//// insert duplicate element
			test.insert(Pair<int, int>(0, 1));
			Assert::AreEqual(static_cast<int>(test.count(Pair<int, int>(0, 1))), 1);
			Assert::AreEqual(static_cast<int>(test.size()), 2);

			//// insert another element
			test.insert(Pair<int, int>(1, 2));
			Assert::AreEqual(static_cast<int>(test.count(Pair<int, int>(1, 2))), 1);
			Assert::AreEqual(static_cast<int>(test.size()), 3);

			//// insert another element
			test.insert(Pair<int, int>(1, -1));
			Assert::AreEqual(static_cast<int>(test.count(Pair<int, int>(1, -1))), 1);
			Assert::AreEqual(static_cast<int>(test.size()), 4);
		}
	};
}