
#include <argtable2.h>
#include <valarray>
#include "GRT_Common/GRTMergeCommandLineData.h"
#include "GRT_Common/GRTTableHeaderV1.h"

GRTMergeCommandLineData::GRTMergeCommandLineData() {
    this->buildFull = 0;
    this->buildPerfect = 0;
    this->megabytesToUse = 512;
    this->moveSortedFiles = 0;

    // Default is version 2, max bits.
    this->tableVersion = 2;
    this->bitsOfOutput = 0;

    this->devDebug = 0;

}

GRTMergeCommandLineData::~GRTMergeCommandLineData() {

}

// Parses the command line.  Returns 0 for failure, 1 for success.
int GRTMergeCommandLineData::ParseCommandLine(int argc, char *argv[]) {

    // For verifying that all files are valid tables.
    GRTTableHeaderV1 TableHeader, TableHeaderTest;
    std::string tableFilename;

    // Output Directory - if not specified, uses /output/ for output.
    struct arg_str *output_directory = arg_str0("o", "outputdir", "<path>", "Directory for output files");
    // Move sorted files directory - if not specified, does not move sorted files.
    struct arg_str *move_sorted_directory = arg_str0(NULL, "movesorted", "<path>", "Directory to move successfully sorted files");

    // Tables to build
    struct arg_lit *build_full = arg_lit0(NULL, "buildfull", "Build a full table");
    struct arg_lit *build_perfect = arg_lit0(NULL, "buildperfect", "Build a perfect table");
    struct arg_int *megabytes_to_use = arg_int0("m", "megabytes", "<megabytes>", "Megabytes of RAM to use");
    // Input files to merge
    struct arg_file *files = arg_filen(NULL,NULL,"<file>",1, 1000000, "table parts to merge");

    // V2 data types
    struct arg_int *table_version = arg_int0(NULL, "tableversion", "<n>", "table version (1 or 2)");
    struct arg_int *bits = arg_int0(NULL, "bits", "<n>", "bits of hash to keep");

    struct arg_lit *devdebug = arg_lit0(NULL, "devdebug", "Developer debug output");

    struct arg_end *end = arg_end(20);

    void *argtable[] = {output_directory,move_sorted_directory,build_full,
        build_perfect,megabytes_to_use,
        files,table_version,bits,devdebug,end};

    int i;


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

    // If neither table is selected, alert the user.
    if (!(build_full->count || build_perfect->count)) {
        printf("You must select a full table (--buildfull), perfect table (--buildperfect),\nor both!\n");
        exit(1);
    }

    if (build_full->count) {
        this->buildFull = 1;
    }
    if (build_perfect->count) {
        this->buildPerfect = 1;
    }
    if (megabytes_to_use->count) {
        this->megabytesToUse = *megabytes_to_use->ival;
    }
    if (table_version->count) {
        if ((*table_version->ival < 1) || (*table_version->ival > 2)) {
            printf("Invalid table version %d!\n", table_version->ival);
            exit(1);
        }
        this->tableVersion = *table_version->ival;
    }

    if (bits->count) {
        this->bitsOfOutput = *bits->ival;
    }

    if (devdebug->count) {
        this->devDebug = 1;
    }



    // Now we get a table header from the first file in the list, and compare the rest of the files to this.
    if (!TableHeader.isValidTable(files->filename[0], -1)) {
        printf("%s is not a valid GRT table!\n", files->filename[0]);
        exit(1);
    }

    // If an output directory is selected, use it.
    if (output_directory->count) {
        this->outputDirectory = output_directory->sval[0];
    } else {
        this->outputDirectory = "output";
    }
    // If we are moving sorted files, set this up.
    if (move_sorted_directory->count) {
        this->moveSortedFiles = 1;
        this->moveSortedFilesDirectory = move_sorted_directory->sval[0];
    }

    // Read in the first table file to compare the rest to.
    TableHeader.readTableHeader(files->filename[0]);
    TableHeader.printTableHeader();

    // Check the rest of the listed table parts.
    // We've already dealt with file 0.
    printf("\n\n");
    for (i = 0; i < files->count; i++) {
        printf("\rChecking Files... %d / %d    ", i, files->count);
        fflush(stdout);
        // Check to ensure the file is valid
        if (!TableHeaderTest.isValidTable(files->filename[i], -1)) {
            printf("%s is not a valid GRT table!\n", files->filename[i]);
            exit(1);
        }

        // Check to see if it matches with the first file loaded
        TableHeaderTest.readTableHeader(files->filename[i]);
        if (!TableHeader.isCompatibleWithTable(&TableHeaderTest)) {
            printf("============= VALIDATION ERROR =============\n");
            printf("%s is not compatible with %s!\n", files->filename[i], files->filename[0]);
            TableHeaderTest.printTableHeader();
            printf("EXITING!\n\n");
            exit(1);
        }
        // Table is good - add it to the list.
        tableFilename = files->filename[i];
        this->filenamesToMerge.push_back(tableFilename);
    }
    printf("\n\n");
    
    printf("Total to merge: %ld\n", this->filenamesToMerge.size());
    return 1;
}

std::vector<std::string> *GRTMergeCommandLineData::getListOfFiles() {
    std::vector<std::string> *returnFiles;

    returnFiles = new std::vector<std::string>();

    *returnFiles = this->filenamesToMerge;
    return returnFiles;
}

char GRTMergeCommandLineData::getBuildPerfect() {
    return this->buildPerfect;
}

char GRTMergeCommandLineData::getBuildFull() {
    return this->buildFull;
}

std::string GRTMergeCommandLineData::getOutputDirectory() {
    return this->outputDirectory;
}

uint32_t GRTMergeCommandLineData::getMegabytesToUse() {
    return this->megabytesToUse;
}

int GRTMergeCommandLineData::getTableVersion() {
    return this->tableVersion;
}

int GRTMergeCommandLineData::getBitsOfOutput() {
    return this->bitsOfOutput;
}

char GRTMergeCommandLineData::getDevDebug() {
    return this->devDebug;
}
std::string GRTMergeCommandLineData::getMoveSortedFilesDirectory() {
    return this->moveSortedFilesDirectory;
}
char GRTMergeCommandLineData::getMoveSortedFiles() {
    return this->moveSortedFiles;
}