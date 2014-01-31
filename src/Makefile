# Builds all the projects in the solution...
.PHONY: all_projects
all_projects: HCSearchLib HCSearch 

# Builds all the projects in the solution with MPI enabled...
.PHONY: mpi
mpi: HCSearchLibMPI HCSearchMPI 

# Builds project 'HCSearchLib'...
.PHONY: HCSearchLib
HCSearchLib: 
	make --directory="HCSearchLib/" --file=HCSearchLib.makefile

# Builds project 'HCSearch'...
.PHONY: HCSearch
HCSearch: HCSearchLib 
	make --directory="HCSearch/" --file=HCSearch.makefile

# Builds project 'HCSearchLib' with MPI enabled...
.PHONY: HCSearchLibMPI
HCSearchLibMPI: 
	make --directory="HCSearchLib/" --file=HCSearchLib.mpi.makefile

# Builds project 'HCSearch' with MPI enabled...
.PHONY: HCSearchMPI
HCSearchMPI: HCSearchLibMPI 
	make --directory="HCSearch/" --file=HCSearch.mpi.makefile

# Cleans all projects...
.PHONY: clean
clean:
	make --directory="HCSearchLib/" --file=HCSearchLib.makefile clean
	make --directory="HCSearch/" --file=HCSearch.makefile clean
	make --directory="HCSearchLib/" --file=HCSearchLib.mpi.makefile clean
	make --directory="HCSearch/" --file=HCSearch.mpi.makefile clean
