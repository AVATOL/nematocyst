# Builds all externals...
.PHONY: all_externals
all_externals: all_required all_optional 

# Builds all required externals...
.PHONY: all_required
all_required: 
	make --directory="external/liblinear/" --file=Makefile
	make --directory="external/svm_rank/" --file=Makefile

# Builds all optional externals...
.PHONY: all_optional
all_optional: 
	make --directory="external/libsvm/" --file=Makefile
	
# Cleans all externals...
.PHONY: clean_externals
clean_externals: clean_required clean_optional 

# Cleans all externals...
.PHONY: clean_required
clean_required:
	make --directory="external/liblinear/" --file=Makefile clean
	make --directory="external/svm_rank/" --file=Makefile clean
	
# Cleans all externals...
.PHONY: clean_optional
clean_optional:
	make --directory="external/libsvm/" --file=Makefile clean
