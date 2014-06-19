# Compiler flags...
CPP_COMPILER = g++
C_COMPILER = gcc

# Include paths...
Debug_Include_Path=
Release_Include_Path=

# Library paths...
Debug_Library_Path=
Release_Library_Path=

# Additional libraries...
Debug_Libraries=
Release_Libraries=

# Preprocessor definitions...
Debug_Preprocessor_Definitions=-D GCC_BUILD -D _DEBUG -D _LIB 
Release_Preprocessor_Definitions=-D GCC_BUILD -D NDEBUG -D _LIB 

# Implictly linked object files...
Debug_Implicitly_Linked_Objects=
Release_Implicitly_Linked_Objects=

# Compiler flags...
Debug_Compiler_Flags=-O0 
Release_Compiler_Flags=-O2 

# Builds all configurations for this project...
.PHONY: build_all_configurations
build_all_configurations: Debug Release 

# Builds the Debug configuration...
.PHONY: Debug
Debug: create_folders gccDebug/DataStructures.o gccDebug/FeatureFunction.o gccDebug/Globals.o gccDebug/HCSearch.o gccDebug/InitialStateFunction.o gccDebug/LossFunction.o gccDebug/MPI.o gccDebug/mtrand.o gccDebug/MyFileSystem.o gccDebug/MyGraphAlgorithms.o gccDebug/MyLogger.o gccDebug/SearchProcedure.o gccDebug/SearchSpace.o gccDebug/Settings.o gccDebug/SuccessorFunction.o 
	ar rcs ../gccDebug/libHCSearchLib.a gccDebug/DataStructures.o gccDebug/FeatureFunction.o gccDebug/Globals.o gccDebug/HCSearch.o gccDebug/InitialStateFunction.o gccDebug/LossFunction.o gccDebug/MPI.o gccDebug/mtrand.o gccDebug/MyFileSystem.o gccDebug/MyGraphAlgorithms.o gccDebug/MyLogger.o gccDebug/SearchProcedure.o gccDebug/SearchSpace.o gccDebug/Settings.o gccDebug/SuccessorFunction.o  $(Debug_Implicitly_Linked_Objects)

# Compiles file DataStructures.cpp for the Debug configuration...
-include gccDebug/DataStructures.d
gccDebug/DataStructures.o: DataStructures.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c DataStructures.cpp $(Debug_Include_Path) -o gccDebug/DataStructures.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM DataStructures.cpp $(Debug_Include_Path) > gccDebug/DataStructures.d

# Compiles file FeatureFunction.cpp for the Debug configuration...
-include gccDebug/FeatureFunction.d
gccDebug/FeatureFunction.o: FeatureFunction.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c FeatureFunction.cpp $(Debug_Include_Path) -o gccDebug/FeatureFunction.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM FeatureFunction.cpp $(Debug_Include_Path) > gccDebug/FeatureFunction.d

# Compiles file Globals.cpp for the Debug configuration...
-include gccDebug/Globals.d
gccDebug/Globals.o: Globals.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c Globals.cpp $(Debug_Include_Path) -o gccDebug/Globals.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM Globals.cpp $(Debug_Include_Path) > gccDebug/Globals.d

# Compiles file HCSearch.cpp for the Debug configuration...
-include gccDebug/HCSearch.d
gccDebug/HCSearch.o: HCSearch.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c HCSearch.cpp $(Debug_Include_Path) -o gccDebug/HCSearch.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM HCSearch.cpp $(Debug_Include_Path) > gccDebug/HCSearch.d

# Compiles file InitialStateFunction.cpp for the Debug configuration...
-include gccDebug/InitialStateFunction.d
gccDebug/InitialStateFunction.o: InitialStateFunction.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c InitialStateFunction.cpp $(Debug_Include_Path) -o gccDebug/InitialStateFunction.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM InitialStateFunction.cpp $(Debug_Include_Path) > gccDebug/InitialStateFunction.d

# Compiles file LossFunction.cpp for the Debug configuration...
-include gccDebug/LossFunction.d
gccDebug/LossFunction.o: LossFunction.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c LossFunction.cpp $(Debug_Include_Path) -o gccDebug/LossFunction.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM LossFunction.cpp $(Debug_Include_Path) > gccDebug/LossFunction.d

# Compiles file MPI.cpp for the Debug configuration...
-include gccDebug/MPI.d
gccDebug/MPI.o: MPI.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c MPI.cpp $(Debug_Include_Path) -o gccDebug/MPI.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM MPI.cpp $(Debug_Include_Path) > gccDebug/MPI.d

# Compiles file mtrand.cpp for the Debug configuration...
-include gccDebug/mtrand.d
gccDebug/mtrand.o: mtrand.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c mtrand.cpp $(Debug_Include_Path) -o gccDebug/mtrand.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM mtrand.cpp $(Debug_Include_Path) > gccDebug/mtrand.d

# Compiles file MyFileSystem.cpp for the Debug configuration...
-include gccDebug/MyFileSystem.d
gccDebug/MyFileSystem.o: MyFileSystem.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c MyFileSystem.cpp $(Debug_Include_Path) -o gccDebug/MyFileSystem.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM MyFileSystem.cpp $(Debug_Include_Path) > gccDebug/MyFileSystem.d

# Compiles file MyGraphAlgorithms.cpp for the Debug configuration...
-include gccDebug/MyGraphAlgorithms.d
gccDebug/MyGraphAlgorithms.o: MyGraphAlgorithms.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c MyGraphAlgorithms.cpp $(Debug_Include_Path) -o gccDebug/MyGraphAlgorithms.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM MyGraphAlgorithms.cpp $(Debug_Include_Path) > gccDebug/MyGraphAlgorithms.d

# Compiles file MyLogger.cpp for the Debug configuration...
-include gccDebug/MyLogger.d
gccDebug/MyLogger.o: MyLogger.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c MyLogger.cpp $(Debug_Include_Path) -o gccDebug/MyLogger.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM MyLogger.cpp $(Debug_Include_Path) > gccDebug/MyLogger.d

# Compiles file SearchProcedure.cpp for the Debug configuration...
-include gccDebug/SearchProcedure.d
gccDebug/SearchProcedure.o: SearchProcedure.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c SearchProcedure.cpp $(Debug_Include_Path) -o gccDebug/SearchProcedure.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM SearchProcedure.cpp $(Debug_Include_Path) > gccDebug/SearchProcedure.d

# Compiles file SearchSpace.cpp for the Debug configuration...
-include gccDebug/SearchSpace.d
gccDebug/SearchSpace.o: SearchSpace.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c SearchSpace.cpp $(Debug_Include_Path) -o gccDebug/SearchSpace.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM SearchSpace.cpp $(Debug_Include_Path) > gccDebug/SearchSpace.d

# Compiles file Settings.cpp for the Debug configuration...
-include gccDebug/Settings.d
gccDebug/Settings.o: Settings.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c Settings.cpp $(Debug_Include_Path) -o gccDebug/Settings.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM Settings.cpp $(Debug_Include_Path) > gccDebug/Settings.d

# Compiles file SuccessorFunction.cpp for the Debug configuration...
-include gccDebug/SuccessorFunction.d
gccDebug/SuccessorFunction.o: SuccessorFunction.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c SuccessorFunction.cpp $(Debug_Include_Path) -o gccDebug/SuccessorFunction.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM SuccessorFunction.cpp $(Debug_Include_Path) > gccDebug/SuccessorFunction.d

# Builds the Release configuration...
.PHONY: Release
Release: create_folders gccRelease/DataStructures.o gccRelease/FeatureFunction.o gccRelease/Globals.o gccRelease/HCSearch.o gccRelease/InitialStateFunction.o gccRelease/LossFunction.o gccRelease/MPI.o gccRelease/mtrand.o gccRelease/MyFileSystem.o gccRelease/MyGraphAlgorithms.o gccRelease/MyLogger.o gccRelease/SearchProcedure.o gccRelease/SearchSpace.o gccRelease/Settings.o gccRelease/SuccessorFunction.o 
	ar rcs ../gccRelease/libHCSearchLib.a gccRelease/DataStructures.o gccRelease/FeatureFunction.o gccRelease/Globals.o gccRelease/HCSearch.o gccRelease/InitialStateFunction.o gccRelease/LossFunction.o gccRelease/MPI.o gccRelease/mtrand.o gccRelease/MyFileSystem.o gccRelease/MyGraphAlgorithms.o gccRelease/MyLogger.o gccRelease/SearchProcedure.o gccRelease/SearchSpace.o gccRelease/Settings.o gccRelease/SuccessorFunction.o  $(Release_Implicitly_Linked_Objects)

# Compiles file DataStructures.cpp for the Release configuration...
-include gccRelease/DataStructures.d
gccRelease/DataStructures.o: DataStructures.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c DataStructures.cpp $(Release_Include_Path) -o gccRelease/DataStructures.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM DataStructures.cpp $(Release_Include_Path) > gccRelease/DataStructures.d

# Compiles file FeatureFunction.cpp for the Release configuration...
-include gccRelease/FeatureFunction.d
gccRelease/FeatureFunction.o: FeatureFunction.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c FeatureFunction.cpp $(Release_Include_Path) -o gccRelease/FeatureFunction.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM FeatureFunction.cpp $(Release_Include_Path) > gccRelease/FeatureFunction.d

# Compiles file Globals.cpp for the Release configuration...
-include gccRelease/Globals.d
gccRelease/Globals.o: Globals.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c Globals.cpp $(Release_Include_Path) -o gccRelease/Globals.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM Globals.cpp $(Release_Include_Path) > gccRelease/Globals.d

# Compiles file HCSearch.cpp for the Release configuration...
-include gccRelease/HCSearch.d
gccRelease/HCSearch.o: HCSearch.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c HCSearch.cpp $(Release_Include_Path) -o gccRelease/HCSearch.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM HCSearch.cpp $(Release_Include_Path) > gccRelease/HCSearch.d

# Compiles file InitialStateFunction.cpp for the Release configuration...
-include gccRelease/InitialStateFunction.d
gccRelease/InitialStateFunction.o: InitialStateFunction.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c InitialStateFunction.cpp $(Release_Include_Path) -o gccRelease/InitialStateFunction.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM InitialStateFunction.cpp $(Release_Include_Path) > gccRelease/InitialStateFunction.d

# Compiles file LossFunction.cpp for the Release configuration...
-include gccRelease/LossFunction.d
gccRelease/LossFunction.o: LossFunction.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c LossFunction.cpp $(Release_Include_Path) -o gccRelease/LossFunction.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM LossFunction.cpp $(Release_Include_Path) > gccRelease/LossFunction.d

# Compiles file MPI.cpp for the Release configuration...
-include gccRelease/MPI.d
gccRelease/MPI.o: MPI.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c MPI.cpp $(Release_Include_Path) -o gccRelease/MPI.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM MPI.cpp $(Release_Include_Path) > gccRelease/MPI.d

# Compiles file mtrand.cpp for the Release configuration...
-include gccRelease/mtrand.d
gccRelease/mtrand.o: mtrand.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c mtrand.cpp $(Release_Include_Path) -o gccRelease/mtrand.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM mtrand.cpp $(Release_Include_Path) > gccRelease/mtrand.d

# Compiles file MyFileSystem.cpp for the Release configuration...
-include gccRelease/MyFileSystem.d
gccRelease/MyFileSystem.o: MyFileSystem.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c MyFileSystem.cpp $(Release_Include_Path) -o gccRelease/MyFileSystem.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM MyFileSystem.cpp $(Release_Include_Path) > gccRelease/MyFileSystem.d

# Compiles file MyGraphAlgorithms.cpp for the Release configuration...
-include gccRelease/MyGraphAlgorithms.d
gccRelease/MyGraphAlgorithms.o: MyGraphAlgorithms.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c MyGraphAlgorithms.cpp $(Release_Include_Path) -o gccRelease/MyGraphAlgorithms.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM MyGraphAlgorithms.cpp $(Release_Include_Path) > gccRelease/MyGraphAlgorithms.d

# Compiles file MyLogger.cpp for the Release configuration...
-include gccRelease/MyLogger.d
gccRelease/MyLogger.o: MyLogger.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c MyLogger.cpp $(Release_Include_Path) -o gccRelease/MyLogger.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM MyLogger.cpp $(Release_Include_Path) > gccRelease/MyLogger.d

# Compiles file SearchProcedure.cpp for the Release configuration...
-include gccRelease/SearchProcedure.d
gccRelease/SearchProcedure.o: SearchProcedure.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c SearchProcedure.cpp $(Release_Include_Path) -o gccRelease/SearchProcedure.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM SearchProcedure.cpp $(Release_Include_Path) > gccRelease/SearchProcedure.d

# Compiles file SearchSpace.cpp for the Release configuration...
-include gccRelease/SearchSpace.d
gccRelease/SearchSpace.o: SearchSpace.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c SearchSpace.cpp $(Release_Include_Path) -o gccRelease/SearchSpace.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM SearchSpace.cpp $(Release_Include_Path) > gccRelease/SearchSpace.d

# Compiles file Settings.cpp for the Release configuration...
-include gccRelease/Settings.d
gccRelease/Settings.o: Settings.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c Settings.cpp $(Release_Include_Path) -o gccRelease/Settings.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM Settings.cpp $(Release_Include_Path) > gccRelease/Settings.d

# Compiles file SuccessorFunction.cpp for the Release configuration...
-include gccRelease/SuccessorFunction.d
gccRelease/SuccessorFunction.o: SuccessorFunction.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c SuccessorFunction.cpp $(Release_Include_Path) -o gccRelease/SuccessorFunction.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM SuccessorFunction.cpp $(Release_Include_Path) > gccRelease/SuccessorFunction.d

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

