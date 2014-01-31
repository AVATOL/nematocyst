# Builds all the projects in the solution...
.PHONY: all_projects
all_projects: HCSearchLib HCSearch 

# Builds project 'HCSearchLib'...
.PHONY: HCSearchLib
HCSearchLib: 
	make --directory="HCSearchLib/" --file=HCSearchLib.makefile

# Builds project 'HCSearch'...
.PHONY: HCSearch
HCSearch: HCSearchLib 
	make --directory="HCSearch/" --file=HCSearch.makefile

# Cleans all projects...
.PHONY: clean
clean:
	make --directory="HCSearchLib/" --file=HCSearchLib.makefile clean
	make --directory="HCSearch/" --file=HCSearch.makefile clean
