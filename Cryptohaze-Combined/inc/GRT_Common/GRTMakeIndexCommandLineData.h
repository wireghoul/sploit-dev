

#include "GRT_Common/GRTCommon.h"

#include <argtable2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>

using namespace std;

// This class gets the command line data options for the index utility
class GRTMakeIndexCommandLineData {
public:
    GRTMakeIndexCommandLineData();
    ~GRTMakeIndexCommandLineData();

    int ParseCommandLine(int argc, char *argv[]);

    int getNumberOfTableFiles();
	std::string getNextTableFile();

    int getBitsToIndex();

private:
    // How many bits of hash to use as the index value
    int bitsToIndex;

    // Vector of filenames to index
    vector<string> filesToIndex;

    int Current_Table_File;   // What number we are currently on

};