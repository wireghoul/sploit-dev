/*
Cryptohaze GPU Rainbow Tables
Copyright (C) 2011  Bitweasil (http://www.cryptohaze.com/)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#ifndef _GRTCLCRACKCOMMANDLINEDATA_H
#define _GRTCLCRACKCOMMANDLINEDATA_H

#include <limits.h>
#include "GRT_Common/GRTHashes.h"
#include "GRT_Common/GRTCommon.h"
#include "GRT_Common/GRTTableHeaderV1.h"
#include "GRT_Common/GRTTableHeaderV2.h"
#include "GRT_Common/GRTTableHeaderVWeb.h"
#include <string>

using namespace std;

#define MIN_PASSWORD_LENGTH 5

#define DEFAULT_KERNEL_TIME 50

#define DEFAULT_WORKITEM_COUNT 0
#define DEFAULT_WORKGROUP_COUNT 0

// Define the default number of candidate hashes to skip for WebTables
#define DEFAULT_CANDIDATES_TO_SKIP 2

class GRTCLCrackCommandLineData {
private:

      int HashType;  // Hash type
      unsigned char Hash[16];

      string hashFileName;
      char useHashFile;
      string outputHashFileName;
      char useOutputHashFile;
        
      char AddHexOutput;
      
      unsigned int KernelTimeMs; // Desired kernel duration, 0 for "run till done"
      unsigned int OpenCLWorkgroups; // Number of blocks to use
      unsigned int OpenCLWorkitems; // Threads per block
      int CUDANumberDevices; // How many CUDA devices are present in the system
      char CUDAUseZeroCopy;

      unsigned int OpenCLDevice; // Device ID
      unsigned int OpenCLPlatform;

      char Autotune; // Set to 1 to autotune the news (thread/block count)
      char CommandLineValid; // Set to 1 if the command line is parsed successfully.
      int Table_File_Count;   // Number of files
      int Current_Table_File;   // What number we are currently on
      char **Table_Filenames; // 2d array

      // Output modifiers
      char Verbose;
      char Silent;

      int useAmdKernels;
      int vectorWidth;

      char Debug;
      char DeveloperDebug;

      // Debug dump: Dump candidate hashes and chains to regen.
      char DebugDump;
      std::string DebugDumpFilenameBase;

      int NumberPrefetchThreads;

      char useWebTable;
      std::string tableURL;
      std::string tableUsername;
      std::string tablePassword;

	  // Candidate hashes to skip
	  int CandidateHashesToSkip;

      GRTHashes *GRTHashTypes;
      GRTTableHeader *TableHeader;
public:
      GRTCLCrackCommandLineData();
    ~GRTCLCrackCommandLineData();

    // Parses the command line.  Returns 0 for failure, 1 for success.
    int ParseCommandLine(int argc, char *argv[]);

    // Getters, all the setting is done in ParseCommandLine
    int getHashType();
    unsigned int getOpenCLDevice();
    unsigned int getOpenCLPlatform();
    unsigned int getKernelTimeMs();
    unsigned int getWorkgroups();
    unsigned int getWorkitems();
    int getUseAmdKernels();
    int getVectorWidth();

    int getNumberOfTableFiles();
    const char *getNextTableFile();

    // Hash parameters
    unsigned char *getHash();
    string getHashFileName();

    char getIsSilent();

    char getUseOutputHashFile();
    string getOutputHashFile();

    int getCudaNumberDevices();

    char GetUseZeroCopy();

    char getUseHashFile();
    
    char getAddHexOutput() {
        return this->AddHexOutput;
    }

    int getNumberPrefetchThreads() {
        return this->NumberPrefetchThreads;
    }

    char getDebug() {
        return this->Debug;
    }

    char getDeveloperDebug() {
        return this->DeveloperDebug;
    }
    
    std::string getTableURL() {
        return this->tableURL;
    }
    std::string getTableUsername() {
        return this->tableUsername;
    }
    std::string getTablePassword() {
        return this->tablePassword;
    }
    char getUseWebTable() {
        return this->useWebTable;
    }
    int getCandidateHashesToSkip() {
            return this->CandidateHashesToSkip;
    }
    char getDebugDump() {
        return this->DebugDump;
    }
    std::string getDebugDumpFilename() {
        return this->DebugDumpFilenameBase;
    }

};


#endif
