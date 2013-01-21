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

// Base class for chain regeneration.  This handles most of the functionality.
// The specific details are handled in the derived classes.

#ifndef __GRTREGENERATECHAINS_H__
#define __GRTREGENERATECHAINS_H__


#include <vector>
#include <algorithm>
#include "GRT_Common/GRTCommon.h"
#include "GRT_CUDA_host/GRTCrackCommandLineData.h"
#include "GRT_Common/GRTTableHeader.h"
#include "CH_Common/GRTWorkunit.h"
#include "CH_Common/GRTHashFilePlain.h"
#include "GRT_Common/GRTCrackDisplay.h"
#include "CUDA_Common/CUDA_Common.h"

#if USE_BOOST_THREADS
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#else
#include <pthread.h>
#endif

using namespace std;


// Maximum supported GPU/CPU thread count.  Updating this will require
// updating the display code to scroll the rates.
#define MAX_SUPPORTED_THREADS 16

// Forward declare CHHashType for the struct
class GRTRegenerateChains;

// Runtime data passed to each CPU/GPU thread.
typedef struct GRTRegenerateThreadRunData {
    char valid;         // Nonzero to indicate a valid unit.
    char cpuThread;     // Nonzero if a CPU thread
    char gpuThread;     // Nonzero if a GPU thread
    char gpuDeviceId;   // CUDA device ID
    int  threadID;      // Thread ID - for identification/structures.
    int  CUDAThreads;   // CUDA thread count for GPUs
    int  CUDABlocks;    // CUDA block count for GPUs
    int  kernelTimeMs;  // Target execution time for GPUs
    GRTRegenerateChains *RegenerateChains;   // Copy of the invoking class to reenter.
 } GRTRegenerateThreadRunData;


class GRTRegenerateChains {
public:

    GRTRegenerateChains(int hashLengthBytes);

    // Add a GPU deviceID to the list of active devices.
    // Returns 0 on failure (probably too many threads), 1 on success.
    int addGPUDeviceID(int deviceId);

    // Add a CPU thread to execute.
    // Returns 0 on failure, 1 on success.
    int addCPUThread();

    // Not needed... HashFile has all of this.
    //int addHashToCrack(hashPasswordData *hashToAdd, int hashLength);

    // Run what we have.
    int regenerateChains();

    // Per-hash functions to be implemented.
    // Copies the bulk of the table data to constant memory for use.
    virtual void copyDataToConstant(GRTRegenerateThreadRunData *data) = 0;
    virtual void setNumberOfChainsToRegen(uint32_t) = 0;


    virtual void Launch_CUDA_Kernel(unsigned char *InitialPasswordArray, unsigned char *FoundPasswordArray,
        unsigned char *DeviceHashArray, UINT4 PasswordSpaceOffset, UINT4 StartChainIndex,
        UINT4 StepsToRun, UINT4 charset_offset, unsigned char *successArray, GRTRegenerateThreadRunData *data) = 0;

    void setCommandLineData(GRTCrackCommandLineData *NewCommandLineData);
    void setTableHeader(GRTTableHeader *NewTableHeader);
    void setWorkunit(GRTWorkunit *NewWorkunit);
    void setHashfile(GRTHashFilePlain *NewHashfile);
    void setDisplay(GRTCrackDisplay *NewDisplay);

    void setChainsToRegen(std::vector<hashPasswordData>*);

    void GPU_Thread(void *);

    void RunGPUWorkunit(GRTWorkunitElement *WU, GRTRegenerateThreadRunData *data);

    // Allocate & free memory for each GPU context
    void AllocatePerGPUMemory(GRTRegenerateThreadRunData *data);
    void FreePerGPUMemory(GRTRegenerateThreadRunData *data);
    int outputFoundHashes(struct GRTRegenerateThreadRunData *data);

protected:
    int HashLengthBytes;
    int PasswordLength;

    // Hash data - constant across all threads.
    unsigned char *HashList;
    uint64_t NumberOfHashes;

    std::vector<hashPasswordData>* ChainsToRegen;

    // Charset data to the GPUs.
    // This gets put in constant memory by the per-hash copyToConstant call.
    char **hostConstantCharset;
    char *hostConstantCharsetLengths;

    // Host and device array pointers for each thread.
    // This stores the pointer to the device hash list (to search for)for each device.
    unsigned char *DEVICE_Hashes[MAX_SUPPORTED_THREADS];

    unsigned char *HOST_Passwords[MAX_SUPPORTED_THREADS];
    unsigned char *DEVICE_Passwords[MAX_SUPPORTED_THREADS];

    unsigned char *HOST_Success[MAX_SUPPORTED_THREADS];
    unsigned char *HOST_Success_Reported[MAX_SUPPORTED_THREADS];
    unsigned char *DEVICE_Success[MAX_SUPPORTED_THREADS];

    // Device start points are generated per-thread due to possible size differences.

    // Keep the per-step count to prevent slowdowns
    uint64_t per_step[MAX_SUPPORTED_THREADS];

    // Bitmaps for lookup - constant across all threads.
    unsigned char hostConstantBitmap[8192];

    // Multithreading data.
    int ActiveThreadCount;
    GRTRegenerateThreadRunData ThreadData[MAX_SUPPORTED_THREADS];

#if USE_BOOST_THREADS
    boost::thread *ThreadObjects[MAX_SUPPORTED_THREADS];
#else
    pthread_t ThreadIds[MAX_SUPPORTED_THREADS];
#endif

    virtual void createConstantBitmap8kb();


    GRTCrackCommandLineData *CommandLineData;
    GRTTableHeader *TableHeader;
    GRTWorkunit *Workunit;
    GRTHashFilePlain *HashFile;
    GRTCrackDisplay *Display;
    char statusStrings[1024];

    // Vector of hashes to search for.
    // The passwords will be filled in as found.
    vector<hashPasswordData> hashesToSearchFor;

};





#endif