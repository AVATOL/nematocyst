# Compiler flags...
CPP_COMPILER = g++
C_COMPILER = gcc

# Include paths...
Debug_Include_Path=-I"../HCSearchLib" -I"UnitTest/include" 
Release_Include_Path=-I"../HCSearchLib" -I"UnitTest/include" 

# Library paths...
Debug_Library_Path=-L"UnitTest/gcclib" 
Release_Library_Path=-L"UnitTest/gcclib" 

# Additional libraries...
Debug_Libraries=-Wl,--start-group -lmsmpi -lmsmpifec -lmsmpifes -lmsmpifmc -lmsmpifms  -Wl,--end-group
Release_Libraries=-Wl,--start-group -lmsmpi -lmsmpifec -lmsmpifes -lmsmpifmc -lmsmpifms  -Wl,--end-group

# Preprocessor definitions...
Debug_Preprocessor_Definitions=-D GCC_BUILD -D _DEBUG 
Release_Preprocessor_Definitions=-D GCC_BUILD -D NDEBUG 

# Implictly linked object files...
Debug_Implicitly_Linked_Objects=
Release_Implicitly_Linked_Objects=

# Compiler flags...
Debug_Compiler_Flags=-fPIC -O0 -g 
Release_Compiler_Flags=-fPIC -O2 

# Builds all configurations for this project...
.PHONY: build_all_configurations
build_all_configurations: Debug Release 

# Builds the Debug configuration...
.PHONY: Debug
Debug: create_folders gccDebug/MyGraphAlgorithmsTests.o gccDebug/SearchSpaceTests.o gccDebug/SettingsTests.o gccDebug/stdafx.o gccDebug/HCSearchTests.o 
	g++ -fPIC -shared -Wl,-soname,libHCSearchTests.so -o ../gccDebug/libHCSearchTests.so gccDebug/MyGraphAlgorithmsTests.o gccDebug/SearchSpaceTests.o gccDebug/SettingsTests.o gccDebug/stdafx.o gccDebug/HCSearchTests.o  $(Debug_Implicitly_Linked_Objects)

# Compiles file MyGraphAlgorithmsTests.cpp for the Debug configuration...
-include gccDebug/MyGraphAlgorithmsTests.d
gccDebug/MyGraphAlgorithmsTests.o: MyGraphAlgorithmsTests.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c MyGraphAlgorithmsTests.cpp $(Debug_Include_Path) -o gccDebug/MyGraphAlgorithmsTests.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM MyGraphAlgorithmsTests.cpp $(Debug_Include_Path) > gccDebug/MyGraphAlgorithmsTests.d

# Compiles file SearchSpaceTests.cpp for the Debug configuration...
-include gccDebug/SearchSpaceTests.d
gccDebug/SearchSpaceTests.o: SearchSpaceTests.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c SearchSpaceTests.cpp $(Debug_Include_Path) -o gccDebug/SearchSpaceTests.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM SearchSpaceTests.cpp $(Debug_Include_Path) > gccDebug/SearchSpaceTests.d

# Compiles file SettingsTests.cpp for the Debug configuration...
-include gccDebug/SettingsTests.d
gccDebug/SettingsTests.o: SettingsTests.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c SettingsTests.cpp $(Debug_Include_Path) -o gccDebug/SettingsTests.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM SettingsTests.cpp $(Debug_Include_Path) > gccDebug/SettingsTests.d

# Compiles file stdafx.cpp for the Debug configuration...
-include gccDebug/stdafx.d
gccDebug/stdafx.o: stdafx.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c stdafx.cpp $(Debug_Include_Path) -o gccDebug/stdafx.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM stdafx.cpp $(Debug_Include_Path) > gccDebug/stdafx.d

# Compiles file HCSearchTests.cpp for the Debug configuration...
-include gccDebug/HCSearchTests.d
gccDebug/HCSearchTests.o: HCSearchTests.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c HCSearchTests.cpp $(Debug_Include_Path) -o gccDebug/HCSearchTests.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM HCSearchTests.cpp $(Debug_Include_Path) > gccDebug/HCSearchTests.d

# Builds the Release configuration...
.PHONY: Release
Release: create_folders gccRelease/MyGraphAlgorithmsTests.o gccRelease/SearchSpaceTests.o gccRelease/SettingsTests.o gccRelease/stdafx.o gccRelease/HCSearchTests.o 
	g++ -fPIC -shared -Wl,-soname,libHCSearchTests.so -o ../gccRelease/libHCSearchTests.so gccRelease/MyGraphAlgorithmsTests.o gccRelease/SearchSpaceTests.o gccRelease/SettingsTests.o gccRelease/stdafx.o gccRelease/HCSearchTests.o  $(Release_Implicitly_Linked_Objects)

# Compiles file MyGraphAlgorithmsTests.cpp for the Release configuration...
-include gccRelease/MyGraphAlgorithmsTests.d
gccRelease/MyGraphAlgorithmsTests.o: MyGraphAlgorithmsTests.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c MyGraphAlgorithmsTests.cpp $(Release_Include_Path) -o gccRelease/MyGraphAlgorithmsTests.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM MyGraphAlgorithmsTests.cpp $(Release_Include_Path) > gccRelease/MyGraphAlgorithmsTests.d

# Compiles file SearchSpaceTests.cpp for the Release configuration...
-include gccRelease/SearchSpaceTests.d
gccRelease/SearchSpaceTests.o: SearchSpaceTests.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c SearchSpaceTests.cpp $(Release_Include_Path) -o gccRelease/SearchSpaceTests.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM SearchSpaceTests.cpp $(Release_Include_Path) > gccRelease/SearchSpaceTests.d

# Compiles file SettingsTests.cpp for the Release configuration...
-include gccRelease/SettingsTests.d
gccRelease/SettingsTests.o: SettingsTests.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c SettingsTests.cpp $(Release_Include_Path) -o gccRelease/SettingsTests.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM SettingsTests.cpp $(Release_Include_Path) > gccRelease/SettingsTests.d

# Compiles file stdafx.cpp for the Release configuration...
-include gccRelease/stdafx.d
gccRelease/stdafx.o: stdafx.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c stdafx.cpp $(Release_Include_Path) -o gccRelease/stdafx.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM stdafx.cpp $(Release_Include_Path) > gccRelease/stdafx.d

# Compiles file HCSearchTests.cpp for the Release configuration...
-include gccRelease/HCSearchTests.d
gccRelease/HCSearchTests.o: HCSearchTests.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c HCSearchTests.cpp $(Release_Include_Path) -o gccRelease/HCSearchTests.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM HCSearchTests.cpp $(Release_Include_Path) > gccRelease/HCSearchTests.d

# Creates the intermediate and output folders for each configuration...
.PHONY: create_folders
create_folders:
	mkdir -p gccDebug
	mkdir -p ../gccDebug
	mkdir -p gccRelease
	mkdir -p ../gccRelease

# Cleans intermediate and output files (objects, libraries, executables)...
.PHONY: clean
clean:
	rm -f gccDebug/*.o
	rm -f gccDebug/*.d
	rm -f ../gccDebug/*.a
	rm -f ../gccDebug/*.so
	rm -f ../gccDebug/*.dll
	rm -f ../gccDebug/*.exe
	rm -f gccRelease/*.o
	rm -f gccRelease/*.d
	rm -f ../gccRelease/*.a
	rm -f ../gccRelease/*.so
	rm -f ../gccRelease/*.dll
	rm -f ../gccRelease/*.exe

