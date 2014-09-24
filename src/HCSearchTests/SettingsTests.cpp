#include "stdafx.h"
#include "CppUnitTest.h"

#include <iostream>
#include "HCSearch.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace HCSearch;

namespace Testing
{		
	TEST_CLASS(SettingsTests)
	{
	public:
		
		TEST_METHOD(ConfigureTest)
		{
			Settings* settings = new Settings();
			settings->refresh("input", "output", "");

			Assert::AreEqual(0, settings->paths->OUTPUT_DIR.compare("output\\"));
			Assert::AreEqual(0, settings->paths->INPUT_DIR.compare("input\\"));
		}
	};
}