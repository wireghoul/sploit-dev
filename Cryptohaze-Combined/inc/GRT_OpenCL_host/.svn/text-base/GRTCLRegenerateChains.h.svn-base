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

#ifndef __GRTCLREGENERATECHAINS_H__
#define __GRTCLREGENERATECHAINS_H__


#include <vector>
#include <algorithm>
#include "GRT_Common/GRTCommon.h"
#include "GRT_OpenCL_host/GRTCLCrackCommandLineData.h"
#include "GRT_Common/GRTTableHeader.h"
#include "CH_Common/GRTWorkunit.h"
#include "CH_Common/GRTHashFilePlain.h"
#include "GRT_Common/GRTCrackDisplay.h"
#include "OpenCL_Common/GRTOpenCL.h"
#include "GRT_OpenCL_host/GRTCLUtils.h"

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
class GRTCLRegenerateChains;

// Runtime data passed to each CPU/GPU thread.
typedef struct GRTCLRegenerateThreadRunData {
    char valid;         // Nonzero to indicate a valid unit.
    char cpuThread;     // Nonzero if a CPU thread
    char gpuThread;     // Nonzero if a GPU thread
    char OpenCLDeviceId;   // OCL device ID
    char OpenCLPlatformId; // OCL platform ID
    int  threadID;      // Thread ID - for identification/structures.
    int  OpenCLWorkitems;   // CUDA thread count for GPUs
    int  OpenCLWorkgroups;    // CUDA block count for GPUs
    int  kernelTimeMs;  // Target execution time for GPUs
    GRTCLRegenerateChains *RegenerateChains;   // Copy of the invoking class to reenter.
 } GRTCLRegenerateThreadRunData;


class GRTCLRegenerateChains {
public:

    GRTCLRegenerateChains(int hashLengthBytes);

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
    void copyDataToConstant(GRTCLRegenerateThreadRunData *data);

    virtual std::vector<std::string> getHashFileName() = 0;
    virtual std::string getHashKernelName() = 0;
    virtual std::string getKernelSourceString() = 0;

    void setCommandLineData(GRTCLCrackCommandLineData *NewCommandLineData);
    void setTableHeader(GRTTableHeader *NewTableHeader);
    void setWorkunit(GRTWorkunit *NewWorkunit);
    void setHashfile(GRTHashFilePlain *NewHashfile);
    void setDisplay(GRTCrackDisplay *NewDisplay);

    void setChainsToRegen(std::vector<hashPasswordData>*);

    void GPU_Thread(void *);

    void RunGPUWorkunit(GRTWorkunitElement *WU, GRTCLRegenerateThreadRunData *data);

    // Allocate & free memory for each GPU context
    void AllocatePerGPUMemory(GRTCLRegenerateThreadRunData *data);
    void FreePerGPUMemory(GRTCLRegenerateThreadRunData *data);
    int outputFoundHashes(struct GRTCLRegenerateThreadRunData *data);

protected:
    int HashLengthBytes;
    int PasswordLength;

    // Hash data - constant across all threads.
    unsigned char *HashList;
    uint64_t NumberOfHashes;

    // Get an OpenCL object for each thread to keep things clean for now.
    // Long term, we will support different contexts/etc.
    CryptohazeOpenCL *OpenCLContexts[MAX_SUPPORTED_THREADS];
    cl_command_queue OpenCLCommandQueue[MAX_SUPPORTED_THREADS];


    std::vector<hashPasswordData>* ChainsToRegen;

    // Charset data to the GPUs.
    // This gets put in constant memory by the per-hash copyToConstant call.
    char **hostConstantCharset;
    char *hostConstantCharsetLengths;

    // Host and device array pointers for each thread.
    // This stores the pointer to the device hash list (to search for)for each device.
    cl_mem DEVICE_Hashes[MAX_SUPPORTED_THREADS];
    cl_mem DEVICE_Bitmap[MAX_SUPPORTED_THREADS];

    unsigned char *HOST_Passwords[MAX_SUPPORTED_THREADS];
    cl_mem DEVICE_Passwords[MAX_SUPPORTED_THREADS];

    unsigned char *HOST_Success[MAX_SUPPORTED_THREADS];
    unsigned char *HOST_Success_Reported[MAX_SUPPORTED_THREADS];
    cl_mem DEVICE_Success[MAX_SUPPORTED_THREADS];

    cl_mem DEVICE_Charset[MAX_SUPPORTED_THREADS];

    // Device start points are generated per-thread due to possible size differences.

    // Keep the per-step count to prevent slowdowns
    uint64_t per_step[MAX_SUPPORTED_THREADS];

    // Bitmaps for lookup - constant across all threads.
    unsigned char hostConstantBitmap[8192];

    // Multithreading data.
    int ActiveThreadCount;
    GRTCLRegenerateThreadRunData ThreadData[MAX_SUPPORTED_THREADS];

#if USE_BOOST_THREADS
    boost::thread *ThreadObjects[MAX_SUPPORTED_THREADS];
#else
    pthread_t ThreadIds[MAX_SUPPORTED_THREADS];
#endif

    virtual void createConstantBitmap8kb();


    GRTCLCrackCommandLineData *CommandLineData;
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