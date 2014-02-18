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

# Builds all the external dependencies...
.PHONY: externals
externals: 
	make --directory="" --file=External.makefile

# Cleans all projects...
.PHONY: clean
clean:
	make --directory="src/HCSearchLib/" --file=HCSearchLib.makefile clean
	make --directory="src/HCSearch/" --file=HCSearch.makefile clean
	make --directory="src/HCSearchLib/" --file=HCSearchLib.mpi.makefile clean
	make --directory="src/HCSearch/" --file=HCSearch.mpi.makefile clean
	rm -f HCSearch

# Cleans all external dependencies...
.PHONY: clean_externals
clean_externals: 
	make --directory="" --file=External.makefile clean_externals
