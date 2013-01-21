#include <vector>
#include <string>
#include "GRT_Common/GRTCommon.h"

using namespace std;

class GRTMergeCommandLineData {
private:
    char buildPerfect;
    char buildFull;
    string outputDirectory;
    string moveSortedFilesDirectory;
    uint32_t megabytesToUse;
    vector<string> filenamesToMerge;

    // Table version data
    int tableVersion;
    int bitsOfOutput;
    char moveSortedFiles;

    // Ultra-verbose debug output
    char devDebug;
    
public:
    GRTMergeCommandLineData();
    ~GRTMergeCommandLineData();

    // Parses the command line.  Returns 0 for failure, 1 for success.
    int ParseCommandLine(int argc, char *argv[]);

    vector<string> *getListOfFiles();
    char getBuildPerfect();
    char getBuildFull();
    string getOutputDirectory();
    uint32_t getMegabytesToUse();

    int getTableVersion();
    int getBitsOfOutput();

    string getMoveSortedFilesDirectory();
    char getMoveSortedFiles();

    char getDevDebug();
};