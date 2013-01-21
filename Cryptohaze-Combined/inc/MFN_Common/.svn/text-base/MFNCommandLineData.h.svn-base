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

#ifndef _MFNCOMMANDLINEDATA_H
#define _MFNCOMMANDLINEDATA_H

#include "Multiforcer_Common/CHCommon.h"

#include <string>
#include <vector>

#define MAX_FILENAME_LENGTH 1024
#define MAX_HOSTNAME_LENGTH 1024


// Current version of the structure - we refuse to restore from different versions.
// This should be incremented any time the structure changes!
#define SAVE_RESTORE_DATA_VERSION 1

/* A structure type containing data for saving and restoring state to the
 * MFNCommandLineData class - this is used for the save/restore files.
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
#pragma pack(push, 1)
typedef struct MFNSaveRestoreData{
    uint8_t  MFNSaveRestoreDataVersion;
    uint8_t  AddHexOutput;
    uint8_t  CurrentPasswordLength;
    uint8_t  UseCharsetMulti;
    int      HashType;
    char     HashListFileName[MAX_FILENAME_LENGTH];
    char     CharsetFileName[MAX_FILENAME_LENGTH];
    char     OutputFileName[MAX_FILENAME_LENGTH];
    char     UnfoundOutputFileName[MAX_FILENAME_LENGTH];
} MFNSaveRestoreData;
#pragma pack(pop)

/**
 * A structure containing device information.  This is used to allow for specifying
 * multiple devices to be added to the execution context.  This structure works
 * for CUDA, OpenCL, and CPU threads.  Unused fields are ignored.
 * 
 * If Block or Thread count is zero, the default for the device is used.
 * 
 * OpenCLPlatformId: The OpenCL Platform ID, if OpenCL is being used.
 * GPUDeviceId: The OpenCL Device ID, or the CUDA Device ID.
 * DeviceBlocks: Number of blocks (CUDA form)
 * DeviceThreads: Number of threads (CUDA form), or number CPU SSE threads.
 * IsCUDADevice: True if this is a CUDA device identifier.
 * IsOpenCLDevice: True if this is an OpenCL device identifier.
 * IsCPUDevice: True if this is a CPU device identifier.
 */
typedef struct MFNDeviceInformation {
    uint32_t OpenCLPlatformId;
    uint32_t GPUDeviceId;
    uint32_t DeviceBlocks;
    uint32_t DeviceThreads;
    uint8_t  IsCUDADevice;
    uint8_t  IsOpenCLDevice;
    uint8_t  IsCPUDevice;
    uint8_t  Reserved1;
} MFNDeviceInformation;

class MFNCommandLineData {
private:
    uint32_t HashType;
    std::string HashTypeString;
    
    std::string HashListFileName;

    std::string CharsetFileName;
    char UseCharsetMulti;

    std::string OutputFileName;

    std::string UnfoundOutputFileName;

    char AddHexOutput;

    // Server mode enabled
    char IsNetworkServer;
    // Server mode - do NOT use GPU/CPU threads - serve only.
    char IsServerOnly;
    // Is a network client
    char IsNetworkClient;

    std::string NetworkRemoteHost;
    uint16_t NetworkPort;


    std::string RestoreFileName;

    // Vector of active devices to use.
    std::vector<MFNDeviceInformation> DevicesToUse;

    uint32_t TargetExecutionTimeMs;

    int DefaultCUDABlocks;
    int DefaultCUDAThreads;

    char UseLookupTable;

    char Verbose;

    char Silent;
    char Daemon;

    char Debug;
    char DevDebug;

    int MinPasswordLength;
    int MaxPasswordLength;

    int WorkunitBits;

    char UseZeroCopy; // Force zero copy memory for integrated GPUs.

    // Force BFI_INT patching on ATI
    char UseBFIIntPatching;
    // Sepecify the OpenCL vector width for kernels
    int VectorWidth;
    // The shared bitmap size - defaults to 8 (kb), and is specified in kb.
    // Valid sizes are 8, 16, 32 - all others are invalid.  Default is 0 (unset)
    int SharedBitmapSize;
    
    // For the hash file class - print the algorithm in use.
    char PrintAlgorithms;
    
public:
    MFNCommandLineData();
    ~MFNCommandLineData();

    // Parses the command line.  Returns 0 for failure, 1 for success.
    int ParseCommandLine(int argc, char *argv[]);


    // Getters, all the setting is done in ParseCommandLine
    virtual uint32_t GetHashType() {
        return this->HashType;
    }
    virtual std::string GetHashTypeString() {
        return this->HashTypeString;
    }

    virtual std::string GetHashListFileName() {
        return this->HashListFileName;
    }

    virtual std::string GetCharsetFileName() {
        return this->CharsetFileName;
    }
    virtual char GetUseCharsetMulti() {
        return this->UseCharsetMulti;
    }

    virtual std::string GetOutputFileName() {
        return this->OutputFileName;
    }

    virtual std::string GetUnfoundOutputFileName() {
        return this->UnfoundOutputFileName;
    }

    virtual std::string GetRestoreFileName() {
        return this->RestoreFileName;
    }

    virtual char GetAddHexOutput() {
        return this->AddHexOutput;
    }
    
    virtual char GetPrintAlgorithms() {
        return this->PrintAlgorithms;
    }

    virtual int GetTargetExecutionTimeMs() {
        return this->TargetExecutionTimeMs;
    }
    
    virtual int GetGpuBlocks() {
        return this->DefaultCUDABlocks;
    }
    
    virtual int GetGpuThreads() {
        return this->DefaultCUDAThreads;
    }

    virtual std::vector<MFNDeviceInformation> GetDevicesToUse() {
        return this->DevicesToUse;
    }

    virtual char GetUseLookupTable() {
        return this->UseLookupTable;
    }

    virtual char GetVerbose() {
        return this->Verbose;
    }
    virtual char GetSilent() {
        return this->Silent;
    }
    virtual char GetDaemon() {
        return this->Daemon;
    }
    virtual char GetDebug() {
        return this->Debug;
    }
    virtual char GetDevDebug() {
        return this->DevDebug;
    }
    virtual int GetMinPasswordLength() {
        return this->MinPasswordLength;
    }
    virtual int GetMaxPasswordLength() {
        return this->MaxPasswordLength;
    }

    virtual char GetUseZeroCopy() {
        return this->UseZeroCopy;
    }
    virtual char GetUseBfiInt() {
        return this->UseBFIIntPatching;
    }
    virtual int GetVectorWidth() {
        return this->VectorWidth;
    }
    int GetSharedBitmapSize() {
        return this->SharedBitmapSize;
    }

    // Returns zero if not set
    virtual int GetWorkunitBits() {
        return this->WorkunitBits;
    }

    virtual char GetIsNetworkServer() {
        return this->IsNetworkServer;
    }
    virtual char GetIsNetworkClient() {
        return this->IsNetworkClient;
    }
    virtual char GetIsServerOnly() {
        return this->IsServerOnly;
    }
    virtual std::string GetNetworkRemoteHostname() {
        return this->NetworkRemoteHost;
    }
    virtual uint16_t GetNetworkPort() {
        return this->NetworkPort;
    }

    std::vector<uint8_t> GetRestoreData(int passLength);
    void SetDataFromRestore(std::vector<uint8_t>);
};


#endif
