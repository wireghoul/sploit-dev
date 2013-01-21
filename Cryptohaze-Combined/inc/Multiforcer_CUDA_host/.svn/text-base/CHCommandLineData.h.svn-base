/*
Cryptohaze Multiforcer & Wordyforcer - low performance GPU password cracking
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

#ifndef _CHCOMMANDLINEDATA_H
#define _CHCOMMANDLINEDATA_H

#include "Multiforcer_Common/CHCommon.h"


#define MAX_FILENAME_LENGTH 1024
#define MAX_HOSTNAME_LENGTH 1024


// Current version of the structure - we refuse to restore from different versions.
// This should be incremented any time the structure changes!
#define SAVE_RESTORE_DATA_VERSION 1

/* A structure type containing data for saving and restoring state to the
 * CHCommandLineData class - this is used for the save/restore files.
 *
 * The "Use" variables are set based on the presence or absence of the
 * related string - they are not stored individually.
 *
 * The current password length is set by the calling code.
 *
 * Things like network server are set on invocation, not by the
 * restore state - this allows easy changing to add the server if not originally
 * enabled in the execution, or changing the server port.
 */
// Fully pack the structure - performance doesn't matter much here.
#pragma pack(push, 1)
typedef struct CHSaveRestoreData{ 
    uint8_t  CHSaveRestoreDataVersion;
    uint8_t  AddHexOutput;
    uint8_t  CurrentPasswordLength;
    uint8_t  UseCharsetMulti;
    int      HashType;
    char     HashListFileName[MAX_FILENAME_LENGTH];
    char     CharsetFileName[MAX_FILENAME_LENGTH];
    char     OutputFileName[MAX_FILENAME_LENGTH];
    char     UnfoundOutputFileName[MAX_FILENAME_LENGTH];
} CHSaveRestoreData;
#pragma pack(pop)

class CHCommandLineData {
private:
    int HashType;

    char HashListFileName[MAX_FILENAME_LENGTH];

    char CharsetFileName[MAX_FILENAME_LENGTH];
    char UseCharsetMulti;

    char OutputFileName[MAX_FILENAME_LENGTH];
    char UseOutputFile;

    char AddHexOutput;

    // Server mode enabled
    char IsNetworkServer;
    // Server mode - do NOT use GPU/CPU threads - serve only.
    char IsServerOnly;
    // Is a network client
    char IsNetworkClient;
    char NetworkRemoteHost[MAX_HOSTNAME_LENGTH];
    uint16_t NetworkPort;

    char UnfoundOutputFileName[MAX_FILENAME_LENGTH];
    char UseUnfoundOutputFile;

    char RestoreFileName[MAX_FILENAME_LENGTH];
    char UseRestoreFile;

    int CUDADevice; // Null if all devices are to be used.
    int CUDANumberDevices; // How many CUDA devices are present in the system
    int TargetExecutionTimeMs;
    int CUDABlocks;
    int CUDAThreads;
    char UseLookupTable;
    char Autotune;

    char Verbose;

    char Silent;
    char Daemon;

    char Debug;
    char DevDebug;

    int MinPasswordLength;
    int MaxPasswordLength;

    int WorkunitBits;

    int UseZeroCopy; // Force zero copy memory for integrated GPUs.

    char CommandLineValid; // Set to 1 if the command line is parsed successfully.

public:
    CHCommandLineData();
    ~CHCommandLineData();

    // Parses the command line.  Returns 0 for failure, 1 for success.
    int ParseCommandLine(int argc, char *argv[]);


    // Getters, all the setting is done in ParseCommandLine
    int GetHashType();

    char* GetHashListFileName();

    char* GetCharsetFileName();
    char GetUseCharsetMulti();

    char* GetOutputFileName();
    char GetUseOutputFile();

    char* GetUnfoundOutputFileName();
    char GetUseUnfoundOutputFile();

    char* GetRestoreFileName();
    char GetUseRestoreFile();

    char GetAddHexOutput();

    int GetCUDADevice();

    int GetCUDANumberDevices();

    int GetTargetExecutionTimeMs();
    // Setters for these, for autotune use.
    void SetCUDABlocks(int CUDABlocks);
    int GetCUDABlocks();
    void SetCUDAThreads(int CUDAThreads);
    int GetCUDAThreads();
    char GetUseLookupTable();
    char GetAutotune();
    char GetVerbose();
    char GetSilent();
    char GetDaemon();
    char GetDebug();
    char GetDevDebug();
    int GetMinPasswordLength();
    int GetMaxPasswordLength();

    int GetUseZeroCopy();
    void SetUseZeroCopy(int);

    // Returns zero if not set
    int GetWorkunitBits();

    char GetIsNetworkServer();
    char GetIsNetworkClient();
    char GetIsServerOnly();
    char *GetNetworkRemoteHostname();
    uint16_t GetNetworkPort();

    std::vector<uint8_t> GetRestoreData(int passLength);
    void SetDataFromRestore(std::vector<uint8_t>);
};


#endif
