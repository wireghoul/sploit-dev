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

#ifndef _GRTGENCOMMANDLINEDATA_H
#define _GRTGENCOMMANDLINEDATA_H

#include <limits.h>
#include "GRT_Common/GRTHashes.h"
#include "GRT_Common/GRTCommon.h"
#include "CUDA_Common/CUDA_Common.h"
#include "CH_Common/CHRandom.h"

#define MAX_FILENAME_LENGTH 1024
//#define MAX_PASSWORD_LENGTH 16
#define MIN_PASSWORD_LENGTH 5

#define CHAIN_LENGTH_WARN_MIN 10000
#define CHAIN_LENGTH_WARN_MAX 1000000

#define DEFAULT_KERNEL_TIME 50
#define DEFAULT_BLOCK_COUNT 16
#define DEFAULT_THREAD_COUNT 128

// Default number of bits in a V3 table
#define DEFAULT_V3_BITS 96

class GRTGenCommandLineData {
private:

      int HashType;  // Hash type
      int PasswordLength; // Length of the passwords to generate
      uint32_t TableIndex;  // Table index
      uint32_t ChainLength; // Length of each chain
      uint32_t NumberChains; // Number of chains to generate
      uint32_t NumberTables;
      uint32_t RandomSeed; // Seed to use for initial password generation
      char CharsetFileName[1024];
      unsigned int CUDADevice; // Device ID
      unsigned int KernelTimeMs; // Desired kernel duration, 0 for "run till done"
      unsigned int CUDABlocks; // Number of blocks to use
      unsigned int CUDAThreads; // Threads per block
      char Autotune; // Set to 1 to autotune the news (thread/block count)
      char CommandLineValid; // Set to 1 if the command line is parsed successfully.

      int OutputTableVersion; // For V1 or V2 tables
      int OutputBits;

      GRTHashes *GRTHashTypes;
      CHRandom *RandomGenerator;

      char useWebGenerate;
      std::string WebGenURL;
      std::string WebGenUsername;
      std::string WebGenPassword;

public:
      GRTGenCommandLineData();
    ~GRTGenCommandLineData();

    // Parses the command line.  Returns 0 for failure, 1 for success.
    int ParseCommandLine(int argc, char *argv[]);

    void PrintTableData();

    // Getters and setters.
    int getHashType() {
        return this->HashType;
    }
    void setHashType(int newHashType) {
        this->HashType = newHashType;
    }

    int getPasswordLength() {
        return this->PasswordLength;
    }
    void setPasswordLength(int newPasswordLength) {
        this->PasswordLength = newPasswordLength;
    }

    uint32_t getTableIndex() {
        return this->TableIndex;
    }
    void setTableIndex(uint32_t newTableIndex) {
        this->TableIndex = newTableIndex;
    }
    
    uint32_t getChainLength() {
        return this->ChainLength;
    }
    void setChainLength(uint32_t newChainLength) {
        this->ChainLength = newChainLength;
    }

    uint32_t getNumberChains() {
        return this->NumberChains;
    }
    void setNumberChains(uint32_t newNumberChains) {
        this->NumberChains = newNumberChains;
    }

    uint32_t getNumberTables() {
        return this->NumberTables;
    }
    void setNumberTables(uint32_t newNumberTables) {
        this->NumberTables = newNumberTables;
    }

    uint32_t getRandomSeed() {
        return this->RandomSeed;
    }
    void setRandomSeed(uint32_t newRandomSeed) {
        this->RandomSeed = newRandomSeed;
    }
    
    char* GetCharsetFileName() {
        return this->CharsetFileName;
    }
    unsigned int getCUDADevice() {
        return this->CUDADevice;
    }
    unsigned int getKernelTimeMs() {
        return this->KernelTimeMs;
    }
    unsigned int getCUDABlocks() {
        return this->CUDABlocks;
    }
    unsigned int getCUDAThreads() {
        return this->CUDAThreads;
    }

    int getOutputTableVersion() {
        return this->OutputTableVersion;
    }
    void setOutputTableVersion(int newOutputTableVersion) {
        this->OutputTableVersion = newOutputTableVersion;
    }

    int getOutputTableBits() {
        return this->OutputBits;
    }
    void setOutputTableBits(int newOutputBits) {
        this->OutputBits = newOutputBits;
    }

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
};


#endif
