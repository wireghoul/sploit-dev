#include "GRT_Common/GRTMakeIndexCommandLineData.h"
#include <string>
using namespace std;



GRTMakeIndexCommandLineData::GRTMakeIndexCommandLineData() {
    this->Current_Table_File = 0;
    this->bitsToIndex = 0;
}

GRTMakeIndexCommandLineData::~GRTMakeIndexCommandLineData() {
    
}


int GRTMakeIndexCommandLineData::ParseCommandLine(int argc, char *argv[]) {

    struct arg_int *bits_to_index = arg_int1("b", "bits", "<n>", "bits to index");
    // Input files to index
    struct arg_file *files = arg_filen(NULL,NULL,"<file>",1, 10000, "tables to index");
    struct arg_end *end = arg_end(20);

    void *argtable[] = {bits_to_index,files,end};

    int i;
    string tableFilename;

    // Get arguments, collect data, check for basic errors.
    if (arg_nullcheck(argtable) != 0) {
      printf("error: insufficient memory\n");
    }
    // Look for errors
    int nerrors = arg_parse(argc,argv,argtable);
    if (nerrors > 0) {
      // Print errors, exit.
      arg_print_errors(stdout,end,argv[0]);
      exit(1);
    }

    // Ready to continue checking.

    if (*bits_to_index->ival > 32 || *bits_to_index->ival < 1) {
        printf("Bits to index must be between 1 and 32!\n");
        exit(1);
    }

    this->bitsToIndex =  *bits_to_index->ival;


    // Check to ensure all are tables.
    for (i = 0; i < files->count; i++) {
        // Check to ensure the file is valid - basic sanity check.
        // More detailed check will happen later.
        if (getTableVersion(files->filename[i]) == -1) {
            printf("%s is not a valid GRT table!\n", files->filename[i]);
            exit(1);
        }
    }

    for (i = 0; i < files->count; i++) {
		
        tableFilename = files->filename[i];
        this->filesToIndex.push_back(tableFilename);
    }
}

int GRTMakeIndexCommandLineData::getNumberOfTableFiles() {
    return this->filesToIndex.size();
}

std::string GRTMakeIndexCommandLineData::getNextTableFile() {
	string filename;

	filename = this->filesToIndex.at(this->Current_Table_File);
    this->Current_Table_File++;

	return filename;
}

int GRTMakeIndexCommandLineData::getBitsToIndex() {
    return this->bitsToIndex;
}