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

#ifndef __GRTCANDIDATEHASHES_H__
#define __GRTCANDIDATEHASHES_H__

// GRTCandidateHashes handles generating the needed candidate hashes for
// a given table.  It requires a copy of the header for data.

#include <vector>
#include <algorithm>
#include "GRT_Common/GRTCommon.h"
#include "CUDA_Common/CUDA_Common.h"
#include "GRT_CUDA_host/GRTCrackCommandLineData.h"
#include "GRT_Common/GRTTableHeader.h"
#include "CH_Common/GRTWorkunit.h"
#include "GRT_Common/GRTCrackDisplay.h"

#if USE_BOOST_THREADS
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#else
#include <pthread.h>
#endif


using namespace std;




// Forward declare CHHashType for the struct
class GRTCandidateHashes;

// Runtime data passed to each CPU/GPU thread.
typedef struct GRTThreadRunData {
    char valid;         // Nonzero to indicate a valid unit.
    char cpuThread;     // Nonzero if a CPU thread
    char gpuThread;     // Nonzero if a GPU thread
    char gpuDeviceId;   // CUDA device ID
    int  threadID;      // Thread ID - for identification/structures.
    int  CUDAThreads;   // CUDA thread count for GPUs
    int  CUDABlocks;    // CUDA block count for GPUs
    int  kernelTimeMs;  // Target execution time for GPUs
    GRTCandidateHashes *CandidateHashes;   // Copy of the invoking class to reenter.
 } GRTThreadRunData;

class GRTCandidateHashes{
public:
    
    GRTCandidateHashes(int hashLengthBytes);

    // Add a new hash to crack.  Takes the universal hashPasswordData struct.
    int addHashToCrack(hashPasswordData *hashToAdd, int hashLength);

    void clearCandidates() {
        this->candidateHashes.clear();
    }

    void clearHashesToCrack() {
        this->hashesToCrack.clear();
    }

    // Add a GPU deviceID to the list of active devices.
    // Returns 0 on failure (probably too many threads), 1 on success.
    int addGPUDeviceID(int deviceId);

    // Add a CPU thread to execute.
    // Returns 0 on failure, 1 on success.
    int addCPUThread();

    // Run what we have.
    int generateCandidateHashes();

    // Get the generated/sorted candidate hashes.
    std::vector<hashData> *getGeneratedCandidates();

    
    // Per-hash functions to be implemented.
    // Copies the bulk of the table data to constant memory for use.
    virtual void copyDataToConstant(GRTThreadRunData *data) = 0;
    // Launches the actual kernel
    virtual void runCandidateHashKernel(int PasswordLength, int CUDA_Blocks, int CUDA_Threads,
        unsigned char *DEVICE_End_Hashes, UINT4 ThreadSpaceOffset, UINT4 StartStep, UINT4 StepsToRun) = 0;
    // Copies the hash into constant memory for the GPU
    virtual void setHashInConstant(unsigned char *hash) = 0;

    void setCommandLineData(GRTCrackCommandLineData *NewCommandLineData);
    void setTableHeader(GRTTableHeader *NewTableHeader);
    void setWorkunit(GRTWorkunit *NewWorkunit);
    void setDisplay(GRTCrackDisplay *NewDisplay);

    void GPU_Thread(void *);

    void RunGPUWorkunit(GRTWorkunitElement *WU, GRTThreadRunData *data);

    // Allocate & free memory for each GPU context
    void AllocatePerGPUMemory(GRTThreadRunData *data);
    void FreePerGPUMemory(GRTThreadRunData *data);

	// Set the number of output chains to skip to avoid
	// hash disclosure in WebTables
	void SetCandidateHashesToSkip(int toSkip) {
		this->NumberOutputChainsToSkip = toSkip;
	}

protected:
    int HashLengthBytes;

	// Skip the last N chains to prevent hash disclosure
	int NumberOutputChainsToSkip;

    // Multithreading data.
    int ActiveThreadCount;
    GRTThreadRunData ThreadData[MAX_SUPPORTED_THREADS];

#if USE_BOOST_THREADS
    boost::thread *ThreadObjects[MAX_SUPPORTED_THREADS];
    boost::mutex addCandidateHashMutexBoost;
#else
    pthread_t ThreadIds[MAX_SUPPORTED_THREADS];

    // Some mutexes...
    pthread_mutex_t addCandidateHashMutex;
    pthread_mutexattr_t addCandidateHashMutexAttr;
#endif

    // For the calculated end hashes.  Start hashes are in constant.
    unsigned char *HOST_End_Hashes[MAX_SUPPORTED_THREADS];
    unsigned char *DEVICE_End_Hashes[MAX_SUPPORTED_THREADS];




    GRTCrackCommandLineData *CommandLineData;
    GRTTableHeader *TableHeader;
    GRTWorkunit *Workunit;
    GRTCrackDisplay *Display;

    char statusStrings[1024];

    // Vector of password/hash data to contain the hashes being submitted
    // The passwords will be filled in as found.
    vector<hashPasswordData> hashesToCrack;

    // Vector of candidate hashes generated.  Does not include the password.
    vector <hashData> candidateHashes;
};


#endif