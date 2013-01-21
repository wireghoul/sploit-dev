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

#include "GRT_CUDA_host/GRTCandidateHashes.h"
#include "CH_Common/GRTWorkunit.h"
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <string.h>
#include "CH_Common/CHHiresTimer.h"
#include "CUDA_Common/CUDA_SAFE_CALL.h"

using namespace std;
typedef uint32_t UINT4;

// Silence all output
extern char silent;

// Entry points for pthreads
extern "C" {
    void *CHHashTypeGPUThread(void *);
}

void *CHHashTypeGPUThread(void * pointer) {
    struct GRTThreadRunData *data;

    data = (GRTThreadRunData *) pointer;

    data->CandidateHashes->GPU_Thread(pointer);

#if USE_BOOST_THREADS
    return NULL;
#else
    pthread_exit(NULL);
#endif
}


GRTCandidateHashes::GRTCandidateHashes(int hashLengthBytes) {
    this->HashLengthBytes = hashLengthBytes;
    // Initialize Multithreading data to all null
    this->ActiveThreadCount = 0;
	this->NumberOutputChainsToSkip = 0;

    this->Display = NULL;

    memset(this->ThreadData, 0, MAX_SUPPORTED_THREADS * sizeof(GRTThreadRunData));
#if USE_BOOST_THREADS
    memset(this->ThreadObjects, 0, MAX_SUPPORTED_THREADS * sizeof(boost::thread *));
#else
    memset(this->ThreadIds, 0, MAX_SUPPORTED_THREADS * sizeof(pthread_t));
    pthread_mutexattr_init(&this->addCandidateHashMutexAttr);
    pthread_mutex_init(&this->addCandidateHashMutex, &this->addCandidateHashMutexAttr);
#endif
    
}

int GRTCandidateHashes::addHashToCrack(hashPasswordData *hashToAdd, int hashLength) {
    this->hashesToCrack.push_back(*hashToAdd);
	return 1;
}

// Add a GPU deviceID to the list of active devices.
// Returns 0 on failure (probably too many threads), 1 on success.
int GRTCandidateHashes::addGPUDeviceID(int deviceId) {

    // Ensure the CommandLineData class has been added.
    // This is likely a programmer error, not a runtime error.
    // Therefore, exit.
    if (!this->CommandLineData) {
        printf("GRTCandidateHashes: Must add command line data first!\n");
        exit(1);
    }

    // If the device ID is invalid, reject.
    // This is likely a runtime error, so just return 0.
    if (deviceId >= this->CommandLineData->getCudaNumberDevices()) {
        sprintf(this->statusStrings, "Invalid device ID %d", deviceId);
        this->Display->addStatusLine(this->statusStrings);
        return 0;
    }

    // If all threads are full, do not add the thread.
    // Again, a runtime error, return 0.
    if (this->ActiveThreadCount >= (MAX_SUPPORTED_THREADS - 1)) {
        sprintf(this->statusStrings, "Too many active threads!");
        this->Display->addStatusLine(this->statusStrings);
        return 0;
    }

    // Set up the GPU thread.
    this->ThreadData[this->ActiveThreadCount].valid = 1;
    this->ThreadData[this->ActiveThreadCount].gpuThread = 1;
    this->ThreadData[this->ActiveThreadCount].gpuDeviceId = deviceId;
    this->ThreadData[this->ActiveThreadCount].threadID =
            this->ActiveThreadCount;
    this->ThreadData[this->ActiveThreadCount].CandidateHashes = this;


    // Set up the device parameters
    // If they are manually forced, set them, else set them to defaults.
    // CUDA Blocks
    if (this->CommandLineData->getCUDABlocks()) {
        this->ThreadData[this->ActiveThreadCount].CUDABlocks =
            this->CommandLineData->getCUDABlocks();
    } else {
        this->ThreadData[this->ActiveThreadCount].CUDABlocks =
            getCudaDefaultBlockCountBySPCount(getCudaStreamProcessorCount(deviceId));
    }

    // CUDA Threads
    if (this->CommandLineData->getCUDAThreads()) {
        this->ThreadData[this->ActiveThreadCount].CUDAThreads =
            this->CommandLineData->getCUDAThreads();
    } else {
        this->ThreadData[this->ActiveThreadCount].CUDAThreads =
            getCudaDefaultThreadCountBySPCount(getCudaStreamProcessorCount(deviceId));
    }

    // Default execution time
    if (this->CommandLineData->getKernelTimeMs()) {
        this->ThreadData[this->ActiveThreadCount].kernelTimeMs =
            this->CommandLineData->getKernelTimeMs();
    } else {
        if (getCudaHasTimeout(deviceId)) {
            this->ThreadData[this->ActiveThreadCount].kernelTimeMs =
                DEFAULT_CUDA_EXECUTION_TIME;
        } else {
            this->ThreadData[this->ActiveThreadCount].kernelTimeMs =
                DEFAULT_CUDA_EXECUTION_TIME_NO_TIMEOUT;
        }
    }

    // Increment the active thread count.
    this->ActiveThreadCount++;

    return 1;
}


int GRTCandidateHashes::generateCandidateHashes() {
    int i;

    // All the memory allocs/etc are per-thread.

    // All global work done.  Onto the per-thread work.
    //printf("Creating %d threads\n", this->ActiveThreadCount);
    // Enter all the threads
    for(i = 0; i < this->ActiveThreadCount; i++) {
#if USE_BOOST_THREADS
        this->ThreadObjects[i] = new boost::thread(CHHashTypeGPUThread, &this->ThreadData[i]);
#else
        pthread_create(&this->ThreadIds[i], NULL, CHHashTypeGPUThread, &this->ThreadData[i] );
#endif
    }
    // Wait for them to come back.
    for(i = 0; i < this->ActiveThreadCount; i++) {
#if USE_BOOST_THREADS
        this->ThreadObjects[i]->join();
#else
        pthread_join(this->ThreadIds[i], NULL);
#endif
    }

    // Sort and unique the workunits.
    if (this->Display) {
        sprintf(this->statusStrings, "CH before merge: %d", this->candidateHashes.size());
        this->Display->addStatusLine(this->statusStrings);
    }

    
    sort(this->candidateHashes.begin(), this->candidateHashes.end(), hashDataSortPredicate);

    this->candidateHashes.erase(
            unique( this->candidateHashes.begin(), this->candidateHashes.end(), hashDataUniquePredicate ),
            this->candidateHashes.end() );

    if (this->Display) {
        sprintf(this->statusStrings, "CH after merge: %d", this->candidateHashes.size());
        this->Display->addStatusLine(this->statusStrings);
    }
    return 1;
}


// This is the GPU thread where we do the per-GPU tasks.
void GRTCandidateHashes::GPU_Thread(void *pointer) {
    struct GRTThreadRunData *data;
    GRTWorkunitElement *WU;

    data = (GRTThreadRunData *) pointer;

    // Set the device.
    cudaSetDevice(data->gpuDeviceId);

    // Enable blocking sync.  This dramatically reduces CPU usage.
    // If zero copy is being used, set DeviceMapHost as well
    if (this->CommandLineData->GetUseZeroCopy()) {
        cudaSetDeviceFlags(cudaDeviceBlockingSync | cudaDeviceMapHost);
    } else {
        cudaSetDeviceFlags(cudaDeviceBlockingSync);
    }

    this->AllocatePerGPUMemory(data);
    this->copyDataToConstant(data);
    cudaThreadSynchronize();

    // I... *think* we're ready to rock!
    // As long as we aren't supposed to exit, keep running.
    while (1) {
        WU = this->Workunit->GetNextWorkunit();
        if (WU == NULL) {
            if (this->Display) {
                sprintf(this->statusStrings, "Thread %d out of WU", data->threadID);
                this->Display->addStatusLine(this->statusStrings);
            }
            break;
        }
        
        this->RunGPUWorkunit(WU, data);
        this->Workunit->SubmitWorkunit(WU);

        if (this->Display) {
            this->Display->setWorkunitsCompleted(this->Workunit->GetNumberOfCompletedWorkunits());
            this->Display->setThreadFractionDone(data->threadID, 0.0);
        }
    }
    if (this->Display) {
        this->Display->setThreadCrackSpeed(data->threadID, 0, 0.0);
    }
    this->FreePerGPUMemory(data);
    // Clean up thread context for subsequent setting of thread ID
    cudaThreadExit();
}

void GRTCandidateHashes::RunGPUWorkunit(GRTWorkunitElement *WU, GRTThreadRunData *data) {
    unsigned char *hash;
    unsigned char *HOST_Return_Hashes;
    UINT4 ChainsCompleted, ThreadSpaceOffset, ThreadSpaceOffsetMax, StartStep, StepsPerInvocation;
    CHHiresTimer kernelTimer, totalTimer;
    UINT4 base_offset, i, j, k;
    
    // For adding them into the list of generated hashes
    hashData candidateHashResult;

    // Alright.  We have a workunit, and we have a hash.  Let's rock!

    // Allocate memory for the hash we are about to crack
    hash = new unsigned char[this->HashLengthBytes];

    // Copy the appropriate hash into the GPU constant memory.
    for (i = 0; i < this->HashLengthBytes; i++) {
        hash[i] = this->hashesToCrack[WU->StartPoint].hash[i];
    }
    this->setHashInConstant(hash);

    // Delete the hash storage and null it.
    delete[] hash;
    hash = NULL;



    memset(&candidateHashResult, 0, sizeof(hashData));
    
    ChainsCompleted = 0;
    ThreadSpaceOffset = 0; // thread = threadid + (ThreadSpaceOffset * NumberOfThreads)
    ThreadSpaceOffsetMax = 0;
    StartStep = 0;
    StepsPerInvocation = 100;

    // Timer for total rate
    totalTimer.start();

    // If kernel time is set to zero, run full chains (headless server).
    if (!this->ThreadData[data->threadID].kernelTimeMs) {
        StepsPerInvocation = this->TableHeader->getChainLength();
    }

    ThreadSpaceOffsetMax = (this->TableHeader->getChainLength() /
            (this->ThreadData[data->threadID].CUDABlocks * this->ThreadData[data->threadID].CUDAThreads));
    ThreadSpaceOffsetMax += 1;
    
    // We need to complete as many steps the chain length
    while (ChainsCompleted <= this->TableHeader->getChainLength()) {
        // Offset in the space (run group)
        ThreadSpaceOffset = (ChainsCompleted / 
                (this->ThreadData[data->threadID].CUDABlocks * this->ThreadData[data->threadID].CUDAThreads));
/*
        if (this->Display) {
            printf("\nRunning group %d of %d\n", ThreadSpaceOffset + 1,
                (this->TableHeader->getChainLength() /
                (this->ThreadData[data->threadID].CUDABlocks * this->ThreadData[data->threadID].CUDAThreads)) + 1);
        }
*/
        StartStep = 0;
        StepsPerInvocation = 100; // Reset this each time for sanity

        // Don't run past the end - if we will go past the end, work fewer steps.
        while (StartStep < this->TableHeader->getChainLength()) {
            if (StartStep + StepsPerInvocation > this->TableHeader->getChainLength()) {
                StepsPerInvocation = this->TableHeader->getChainLength() - StartStep;
            }

            kernelTimer.start();

            this->runCandidateHashKernel(this->TableHeader->getPasswordLength(), this->ThreadData[data->threadID].CUDABlocks,
                this->ThreadData[data->threadID].CUDAThreads,
                this->DEVICE_End_Hashes[data->threadID], ThreadSpaceOffset, StartStep, StepsPerInvocation);

            cudaThreadSynchronize();
            cudaError_t err = cudaGetLastError();
            if( cudaSuccess != err)
            {
              fprintf(stderr, "Cuda error: %s.\n", cudaGetErrorString( err) );
              exit(EXIT_FAILURE);
            }

            // cuGetTimerValue returns milliseconds - keep the same to avoid bugs.
            float ref_time = kernelTimer.getElapsedTimeInMilliSec();

            if (this->Display) {
                // Calculate our total trip through the space
                float threadProgress = 0.0;


                if (ThreadSpaceOffsetMax == 1) {
                    // If there is only one workgroup, it's the same as our percentage.
                    threadProgress = (float)(StartStep) / (float)this->TableHeader->getChainLength();
                    this->Display->setThreadFractionDone(data->threadID, threadProgress);
                } else {
                    // This gets more complex.
                    
                    // First, sort out "how far through the groups" we are.
                    threadProgress += ((float)ThreadSpaceOffset / (float)ThreadSpaceOffsetMax);
                    
                    // Next, add our current progress, divided by the number of groups.
                    threadProgress += ((float)(StartStep) / (float)this->TableHeader->getChainLength())
                            / (float)ThreadSpaceOffsetMax;
                    this->Display->setThreadFractionDone(data->threadID, threadProgress);
                }

                /*
                printf("Kernel Time: %0.3f ms  Done: %0.2f%%         \r",ref_time,
                    (float)(100.0 * (float)(StartStep) / (float)this->TableHeader->getChainLength()));
                fflush(stdout);
                */
            }
            StartStep += StepsPerInvocation;

            if (this->ThreadData[data->threadID].kernelTimeMs) {
                // Adjust the steps per invocation if needed.
                if ((ref_time > 1.1 * (float)this->ThreadData[data->threadID].kernelTimeMs) ||
                        (ref_time < 0.9 * (float)this->ThreadData[data->threadID].kernelTimeMs)) {
                    StepsPerInvocation = (UINT4)((float)StepsPerInvocation *
                            ((float)this->ThreadData[data->threadID].kernelTimeMs / ref_time));

                    if (StepsPerInvocation == 0) {
                        StepsPerInvocation = 10;
                    }
                }
            }
        }
 
        // Add the full block size to the number completed.
        ChainsCompleted += this->ThreadData[data->threadID].CUDABlocks * this->ThreadData[data->threadID].CUDAThreads;
    }
 

    float total_time = totalTimer.getElapsedTimeInMilliSec();
    if (this->Display) {
        this->Display->setThreadCrackSpeed(data->threadID, GPU_THREAD,
            (float)((float)this->TableHeader->getChainLength() * (float)this->TableHeader->getChainLength() / 2) / (total_time * 1000.0));
    }


    /*    if (!silent) {
        printf("\n\n");
        printf("Total time: %0.2f seconds\n", total_time / 1000.0);
        printf("Average rate: %0.2f M h/s\n",
            (float)((float)this->TableHeader->getChainLength() * (float)this->TableHeader->getChainLength() / 2) / (total_time * 1000.0));
    }
*/
    cudaMemcpy(this->HOST_End_Hashes[data->threadID], this->DEVICE_End_Hashes[data->threadID],
        this->HashLengthBytes * this->TableHeader->getChainLength() * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Decoalesce the hashes so we have something sane
    // This can *probably* go straight into candidateHashResult...
    HOST_Return_Hashes = new unsigned char[this->HashLengthBytes * this->TableHeader->getChainLength()];

    for (i = 0; i < this->TableHeader->getChainLength(); i++) {
        // Segment
        for (j = 0; j < (this->HashLengthBytes / 4); j++) {
            base_offset = this->TableHeader->getChainLength() * j * 4; // Segment start
            base_offset += i * 4; // This chain start
            for (k = 0; k < 4; k++) {
                HOST_Return_Hashes[i * this->HashLengthBytes + (j * 4) + k] = this->HOST_End_Hashes[data->threadID][base_offset + k];
            }
        }
    }

    // Lock around the whole section to prevent STL thread issues.
#if USE_BOOST_THREADS
    this->addCandidateHashMutexBoost.lock();
#else
    pthread_mutex_lock(&this->addCandidateHashMutex);
#endif
	if (this->Display && this->NumberOutputChainsToSkip) {
        sprintf(this->statusStrings, "Skipping last %d", this->NumberOutputChainsToSkip);
        this->Display->addStatusLine(this->statusStrings);
    }

    // Add the de-interleaved hashes into the output
    for (i = 0; i < (this->TableHeader->getChainLength() - NumberOutputChainsToSkip); i++) {
        // Segment
        for (j = 0; j < (this->HashLengthBytes); j++) {
            candidateHashResult.hash[j] = HOST_Return_Hashes[i * this->HashLengthBytes + j];
        }
        this->candidateHashes.push_back(candidateHashResult);
    }

#if USE_BOOST_THREADS
    this->addCandidateHashMutexBoost.unlock();
#else
    pthread_mutex_unlock(&this->addCandidateHashMutex);
#endif

	// If the developer debug is being used, dump CH
	if (this->CommandLineData->getDeveloperDebug()) {
		for (i = 0; i < this->candidateHashes.size(); i++) {
			printf("Candidate hash %d: ", i);
			for (j = 0; j < this->HashLengthBytes; j++) {
				printf("%02X", this->candidateHashes.at(i).hash[j]);
			}
			printf("\n");
		}
	}

	delete[] HOST_Return_Hashes;
}

// Get the generated/sorted candidate hashes.
std::vector<hashData> *GRTCandidateHashes::getGeneratedCandidates() {
    uint64_t i;
    hashData transferData;
    
    std::vector<hashData> * returnVector = new std::vector<hashData>();
    
    for (i = 0; i < this->candidateHashes.size(); i++) {
        transferData = this->candidateHashes[i];
        returnVector->push_back(transferData);
    }
    
    //printf("%d vectors copied.\n", returnVector->size());

    return returnVector;
}

void GRTCandidateHashes::setCommandLineData(GRTCrackCommandLineData *NewCommandLineData) {
    this->CommandLineData = NewCommandLineData;
}

void GRTCandidateHashes::setTableHeader(GRTTableHeader *NewTableHeader) {
    this->TableHeader = NewTableHeader;
}
void GRTCandidateHashes::setWorkunit(GRTWorkunit *NewWorkunit) {
    this->Workunit = NewWorkunit;
}

void GRTCandidateHashes::setDisplay(GRTCrackDisplay *NewDisplay) {
    this->Display = NewDisplay;
}



void GRTCandidateHashes::AllocatePerGPUMemory(GRTThreadRunData *data) {
    // Default flags are for memory on device and copying things.
    unsigned int flags = 0;


    // If we are using zero copy, set the zero copy flag.
    if (this->CommandLineData->GetUseZeroCopy()) {
        flags = cudaHostAllocMapped;
    }

    // Allocate host memory for the end hashes.
    cudaHostAlloc((void **)&this->HOST_End_Hashes[data->threadID],
        this->HashLengthBytes * this->TableHeader->getChainLength() * sizeof(unsigned char), flags);

    // If zero copy is in use, get the device pointer, else put data on the device.
    if (this->CommandLineData->GetUseZeroCopy()) {
        cudaHostGetDevicePointer((void **)&this->DEVICE_End_Hashes[data->threadID],
            this->HOST_End_Hashes[data->threadID], 0);
    } else {
        CH_CUDA_SAFE_CALL(cudaMalloc((void **)&this->DEVICE_End_Hashes[data->threadID],
            this->HashLengthBytes * this->TableHeader->getChainLength() * sizeof(unsigned char)));
        CH_CUDA_SAFE_CALL(cudaMemset(this->DEVICE_End_Hashes[data->threadID], 0,
            this->HashLengthBytes * this->TableHeader->getChainLength() * sizeof(unsigned char)));
    }
    //printf("Memory for thread %d allocated.\n", data->threadID);
}

void GRTCandidateHashes::FreePerGPUMemory(GRTThreadRunData *data) {
    CH_CUDA_SAFE_CALL(cudaFreeHost(this->HOST_End_Hashes[data->threadID]));
    // Only free the device memory if zero copy was NOT used
    if (!this->CommandLineData->GetUseZeroCopy()) {
        CH_CUDA_SAFE_CALL(cudaFree(this->DEVICE_End_Hashes[data->threadID]));
    }
    //printf("Memory for thread %d freed.\n", data->threadID);
}
