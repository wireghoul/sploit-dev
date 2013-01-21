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

#ifndef _GRTCLGENCOMMANDLINEDATA_H
#define _GRTCLGENCOMMANDLINEDATA_H

#include <limits.h>
#include "GRT_Common/GRTHashes.h"
#include "GRT_Common/GRTCommon.h"
#include "CH_Common/CHRandom.h"


#define MAX_FILENAME_LENGTH 1024
//#define MAX_PASSWORD_LENGTH 16
#define MIN_PASSWORD_LENGTH 5

#define CHAIN_LENGTH_WARN_MIN 10000
#define CHAIN_LENGTH_WARN_MAX 1000000

#define DEFAULT_KERNEL_TIME 50
#define DEFAULT_WORKITEM_COUNT 0
#define DEFAULT_WORKGROUP_COUNT 0

// Default number of bits in a V3 table
#define DEFAULT_V3_BITS 96

class GRTCLGenCommandLineData {
private:

      int HashType;  // Hash type
      int PasswordLength; // Length of the passwords to generate
      uint32_t TableIndex;  // Table index
      uint32_t ChainLength; // Length of each chain
      uint32_t NumberChains; // Number of chains to generate
      uint32_t NumberTables;
      uint32_t RandomSeed; // Seed to use for initial password generation
      char CharsetFileName[1024];
      unsigned int OpenCLDevice; // Device ID
      unsigned int OpenCLPlatform;
      unsigned int KernelTimeMs; // Desired kernel duration, 0 for "run till done"
      unsigned int OpenCLWorkgroups; // Number of blocks to use
      unsigned int OpenCLWorkitems; // Threads per block
      char Autotune; // Set to 1 to autotune the news (thread/block count)
      char CommandLineValid; // Set to 1 if the command line is parsed successfully.

      int OutputTableVersion; // For V1 or V2 tables
      int OutputBits;

      int useAmdKernels;
      int vectorWidth;

      GRTHashes *GRTHashTypes;
      CHRandom *RandomGenerator;

      char useWebGenerate;
      std::string WebGenURL;
      std::string WebGenUsername;
      std::string WebGenPassword;
public:
      GRTCLGenCommandLineData();
    ~GRTCLGenCommandLineData();

    // Parses the command line.  Returns 0 for failure, 1 for success.
    int ParseCommandLine(int argc, char *argv[]);

    void PrintTableData();

    // Getters, all the setting is done in ParseCommandLine
    int getHashType();
    int getPasswordLength();
    uint32_t getTableIndex();
    uint32_t getChainLength();
    uint32_t getNumberChains();
    void setNumberChains(uint32_t);
    uint32_t getNumberTables();
    uint32_t getRandomSeed();
    char* GetCharsetFileName();
    unsigned int getOpenCLDevice();
    unsigned int getOpenCLPlatform();
    unsigned int getKernelTimeMs();
    unsigned int getWorkgroups();
    unsigned int getWorkitems();
    int getUseAmdKernels();
    int getVectorWidth();

    int getOutputTableVersion();
    int getOutputTableBits();

    void setRandomGenerator(CHRandom *newRandom) {
        this->RandomGenerator = newRandom;
    }

    std::string getWebGenURL() {
        return this->WebGenURL;
    }
    std::string getWebUsername() {
        return this->WebGenUsername;
    }
    std::string getWebPassword() {
        return this->WebGenPassword;
    }
    char getUseWebGenerate() {
        return this->useWebGenerate;
    }
    
    void setHashType(int newHashType) {
        this->HashType = newHashType;
    }
    
    void setNumberTables(uint32_t newNumberTables) {
        this->NumberTables = newNumberTables;
    }

};


#endif
