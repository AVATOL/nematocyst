### Main Project ###

# Builds all the projects in the solution...
.PHONY: all_projects
all_projects: HCSearchLib HCSearch 

# Builds all the projects in the solution with MPI enabled...
.PHONY: mpi
mpi: HCSearchLibMPI HCSearchMPI 

# Builds project 'HCSearchLib'...
.PHONY: HCSearchLib
HCSearchLib: 
	make --directory="src/HCSearchLib/" --file=HCSearchLib.makefile

# Builds project 'HCSearch'...
.PHONY: HCSearch
HCSearch: HCSearchLib 
	make --directory="src/HCSearch/" --file=HCSearch.makefile
	cp src/gccRelease/HCSearch .

# Builds project 'HCSearchLib' with MPI enabled...
.PHONY: HCSearchLibMPI
HCSearchLibMPI: 
	make --directory="src/HCSearchLib/" --file=HCSearchLib.mpi.makefile

# Builds project 'HCSearch' with MPI enabled...
.PHONY: HCSearchMPI
HCSearchMPI: HCSearchLibMPI 
	make --directory="src/HCSearch/" --file=HCSearch.mpi.makefile
	cp src/gccRelease/HCSearch .

# Cleans all projects...
.PHONY: clean
clean:
	make --directory="src/HCSearchLib/" --file=HCSearchLib.makefile clean
	make --directory="src/HCSearch/" --file=HCSearch.makefile clean
	make --directory="src/HCSearchLib/" --file=HCSearchLib.mpi.makefile clean
	make --directory="src/HCSearch/" --file=HCSearch.mpi.makefile clean
	rm -f HCSearch

### Externals ###

# Builds all externals...
.PHONY: all_externals
all_externals: externals optional_externals 

# Builds all required externals...
.PHONY: externals
externals: 
	make --directory="external/liblinear/" --file=Makefile
	make --directory="external/svm_rank/" --file=Makefile

# Builds all optional externals...
.PHONY: optional_externals
optional_externals: 
	make --directory="external/libsvm/" --file=Makefile
	
# Cleans all externals...
.PHONY: clean_all_externals
clean_all_externals: clean_externals clean_optional_externals 

# Cleans all externals...
.PHONY: clean_externals 
clean_externals: 
	make --directory="external/liblinear/" --file=Makefile clean
	make --directory="external/svm_rank/" --file=Makefile clean
	
# Cleans all externals...
.PHONY: clean_optional_externals
clean_optional_externals: 
	make --directory="external/libsvm/" --file=Makefile clean
