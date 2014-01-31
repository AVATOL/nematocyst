# Compiler flags...
CPP_COMPILER = mpic++
C_COMPILER = gcc

# Include paths...
Debug_Include_Path=-I"../HCSearchLib" 
Release_Include_Path=-I"../HCSearchLib" 

# Library paths...
Debug_Library_Path=-L"../gccDebug" 
Release_Library_Path=-L"../gccRelease" 

# Additional libraries...
Debug_Libraries=-Wl,--start-group -lHCSearchLib  -Wl,--end-group
Release_Libraries=-Wl,--start-group -lHCSearchLib  -Wl,--end-group

# Preprocessor definitions...
Debug_Preprocessor_Definitions=-D GCC_BUILD -D _DEBUG -D _CONSOLE -D USE_MPI 
Release_Preprocessor_Definitions=-D GCC_BUILD -D NDEBUG -D _CONSOLE -D USE_MPI 

# Implictly linked object files...
Debug_Implicitly_Linked_Objects=
Release_Implicitly_Linked_Objects=

# Compiler flags...
Debug_Compiler_Flags=-O0 -g 
Release_Compiler_Flags=-O2 

# Builds all configurations for this project...
.PHONY: build_all_configurations
build_all_configurations: Debug Release 

# Builds the Debug configuration...
.PHONY: Debug
Debug: create_folders gccDebug/Main.o gccDebug/MyProgramOptions.o 
	mpic++ gccDebug/Main.o gccDebug/MyProgramOptions.o  $(Debug_Library_Path) $(Debug_Libraries) -Wl,-rpath,./ -o ../gccDebug/HCSearch

# Compiles file Main.cpp for the Debug configuration...
-include gccDebug/Main.d
gccDebug/Main.o: Main.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c Main.cpp $(Debug_Include_Path) -o gccDebug/Main.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM Main.cpp $(Debug_Include_Path) > gccDebug/Main.d

# Compiles file MyProgramOptions.cpp for the Debug configuration...
-include gccDebug/MyProgramOptions.d
gccDebug/MyProgramOptions.o: MyProgramOptions.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c MyProgramOptions.cpp $(Debug_Include_Path) -o gccDebug/MyProgramOptions.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM MyProgramOptions.cpp $(Debug_Include_Path) > gccDebug/MyProgramOptions.d

# Builds the Release configuration...
.PHONY: Release
Release: create_folders gccRelease/Main.o gccRelease/MyProgramOptions.o 
	mpic++ gccRelease/Main.o gccRelease/MyProgramOptions.o  $(Release_Library_Path) $(Release_Libraries) -Wl,-rpath,./ -o ../gccRelease/HCSearch

# Compiles file Main.cpp for the Release configuration...
-include gccRelease/Main.d
gccRelease/Main.o: Main.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c Main.cpp $(Release_Include_Path) -o gccRelease/Main.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM Main.cpp $(Release_Include_Path) > gccRelease/Main.d

# Compiles file MyProgramOptions.cpp for the Release configuration...
-include gccRelease/MyProgramOptions.d
gccRelease/MyProgramOptions.o: MyProgramOptions.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c MyProgramOptions.cpp $(Release_Include_Path) -o gccRelease/MyProgramOptions.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM MyProgramOptions.cpp $(Release_Include_Path) > gccRelease/MyProgramOptions.d

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
	rm -f ../gccDebug/HCSearch
	rm -f gccRelease/*.o
	rm -f gccRelease/*.d
	rm -f ../gccRelease/*.a
	rm -f ../gccRelease/*.so
	rm -f ../gccRelease/*.dll
	rm -f ../gccRelease/*.exe
	rm -f ../gccRelease/HCSearch

