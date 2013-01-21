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

#include "GRT_CUDA_host/GRTRegenerateChains.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <string.h>
#include "CUDA_Common/CUDA_SAFE_CALL.h"
#include "CH_Common/CHHiresTimer.h"

extern char silent;

// Entry points for pthreads
extern "C" {
    void *GRTRegenChainGPUThread(void *);
}

void *GRTRegenChainGPUThread(void * pointer) {
    struct GRTRegenerateThreadRunData *data;

    data = (GRTRegenerateThreadRunData *) pointer;

    //printf("IN THREAD %d\n", data->threadID);
    data->RegenerateChains->GPU_Thread(pointer);
    //printf("Thread %d Back from GPU_Thread\n", data->threadID);
    //fflush(stdout);
#if USE_BOOST_THREADS
    return NULL;
#else
    pthread_exit(NULL);
#endif
}


GRTRegenerateChains::GRTRegenerateChains(int hashLengthBytes) {
    this->HashLengthBytes = hashLengthBytes;
    // Initialize Multithreading data to all null
    this->ActiveThreadCount = 0;
    memset(this->ThreadData, 0, MAX_SUPPORTED_THREADS * sizeof(GRTRegenerateThreadRunData));
#if USE_BOOST_THREADS
    memset(this->ThreadObjects, 0, MAX_SUPPORTED_THREADS * sizeof(boost::thread *));
#else
    memset(this->ThreadIds, 0, MAX_SUPPORTED_THREADS * sizeof(pthread_t));
#endif
}


// Add a GPU deviceID to the list of active devices.
// Returns 0 on failure (probably too many threads), 1 on success.
int GRTRegenerateChains::addGPUDeviceID(int deviceId) {

    // Ensure the CommandLineData class has been added.
    // This is likely a programmer error, not a runtime error.
    // Therefore, exit.
    if (!this->CommandLineData) {
        printf("GRTRegenerateChains: Must add command line data first!\n");
        exit(1);
    }

    // If the device ID is invalid, reject.
    // This is likely a runtime error, so just return 0.
    if (deviceId >= this->CommandLineData->getCudaNumberDevices()) {
        if (this->Display) {
            sprintf(this->statusStrings, "Invalid device ID %d", deviceId);
            this->Display->addStatusLine(this->statusStrings);
        } else {
            printf("Invalid device ID %d\n", deviceId);
        }
        return 0;
    }

    // If all threads are full, do not add the thread.
    // Again, a runtime error, return 0.
    if (this->ActiveThreadCount >= (MAX_SUPPORTED_THREADS - 1)) {
        if (this->Display) {
            sprintf(this->statusStrings, "Too many active threads!");
            this->Display->addStatusLine(this->statusStrings);
        } else {
            printf("Too many active threads!\n");
        }
        return 0;
    }

    // Set up the GPU thread.
    this->ThreadData[this->ActiveThreadCount].valid = 1;
    this->ThreadData[this->ActiveThreadCount].gpuThread = 1;
    this->ThreadData[this->ActiveThreadCount].gpuDeviceId = deviceId;
    this->ThreadData[this->ActiveThreadCount].threadID =
            this->ActiveThreadCount;
    this->ThreadData[this->ActiveThreadCount].RegenerateChains = this;


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

    //printf("Added GPU device %d\n", deviceId);

    return 1;
}


int GRTRegenerateChains::regenerateChains() {
    int i;


    // Do the global work
    this->HashList = this->HashFile->ExportUncrackedHashList();
    this->NumberOfHashes = this->HashFile->GetUncrackedHashCount();

    this->hostConstantCharsetLengths = this->TableHeader->getCharsetLengths();
    this->hostConstantCharset = this->TableHeader->getCharset();

    this->PasswordLength = this->TableHeader->getPasswordLength();

    this->createConstantBitmap8kb();

    if (!silent) {
        if (this->Display) {
            sprintf(this->statusStrings, "Num Hashes: %d", this->NumberOfHashes);
            this->Display->addStatusLine(this->statusStrings);
        } else {
            printf("Number of hashes to search for: %d\n", this->NumberOfHashes);
        }
    }


    // All the memory allocs/etc are per-thread.

    // All global work done.  Onto the per-thread work.
    //printf("Creating %d threads\n", this->ActiveThreadCount);
    // Enter all the threads
    for(i = 0; i < this->ActiveThreadCount; i++) {
#if USE_BOOST_THREADS
        this->ThreadObjects[i] = new boost::thread(&GRTRegenChainGPUThread, &this->ThreadData[i]);
#else
        pthread_create(&this->ThreadIds[i], NULL, GRTRegenChainGPUThread, &this->ThreadData[i] );
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
    //printf("Threads joined\n");
	return 1;
}



// This is the GPU thread where we do the per-GPU tasks.
void GRTRegenerateChains::GPU_Thread(void *pointer) {
    struct GRTRegenerateThreadRunData *data;
    GRTWorkunitElement *WU;

    data = (GRTRegenerateThreadRunData *) pointer;

    // Set the device.
    cudaSetDevice(data->gpuDeviceId);

    // Enable blocking sync.  This dramatically reduces CPU usage.
    // If zero copy is being used, set DeviceMapHost as well
    if (this->CommandLineData->GetUseZeroCopy()) {
        cudaSetDeviceFlags(cudaDeviceBlockingSync | cudaDeviceMapHost);
    } else {
        cudaSetDeviceFlags(cudaDeviceBlockingSync);
    }

    //printf("Copying to GPU mem\n");
    this->AllocatePerGPUMemory(data);
    this->copyDataToConstant(data);
   // printf("Back from copy constant\n");
    cudaThreadSynchronize();

    // I... *think* we're ready to rock!
    // As long as we aren't supposed to exit, keep running.
    while (1) {
        WU = this->Workunit->GetNextWorkunit();
        if (WU == NULL) {
            if (!silent) {
                if (this->Display) {
                    sprintf(this->statusStrings, "Thread %d out of WU", data->threadID);
                    this->Display->addStatusLine(this->statusStrings);
                } else {
                    printf("Thread %d out of workunits\n", data->threadID);
                }
            }
            break;
        }
        if (this->HashFile->GetUncrackedHashCount() == 0) {
            if (!silent) {
                if (this->Display) {
                    sprintf(this->statusStrings, "All hashes found");
                    this->Display->addStatusLine(this->statusStrings);
                } else {
                    printf("Thread %d no unfound hashes left!\n", data->threadID);
                }
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




void GRTRegenerateChains::RunGPUWorkunit(GRTWorkunitElement *WU, GRTRegenerateThreadRunData *data) {
    //printf("In RunGPUWorkunit!\n");
    //printf("Got startPoint %d\n", WU->StartPoint);
    //printf("Got endpoint %d\n", WU->EndPoint);
    //printf("Chains to regen: %d\n", WU->EndPoint - WU->StartPoint);


    unsigned char *DEVICE_Chains_To_Regen, *HOST_Interleaved_Chains_To_Regen;
    UINT4 i, j, k;
    UINT4 ChainsCompleted, CurrentChainStartOffset, PasswordSpaceOffset, CharsetOffset, StepsPerInvocation;
    int ActiveThreads;
    CHHiresTimer kernelTimer;

    // Calculate the number of chains being regen'd by this thread.
    uint32_t NumberChainsToRegen = WU->EndPoint - WU->StartPoint + 1;

    this->setNumberOfChainsToRegen(NumberChainsToRegen);

    // Allocate device memory for chains.
    if (cudaErrorMemoryAllocation ==
            cudaMalloc((void**)&DEVICE_Chains_To_Regen, this->HashLengthBytes *
                NumberChainsToRegen * sizeof(unsigned char))) {
        printf("ERROR: Cannot allocate GPU memory.  Try rebooting?\n");
        exit(1);
    }

    // Interleave chains for better GPU performance and coalescing.
    HOST_Interleaved_Chains_To_Regen = (unsigned char *)malloc(this->HashLengthBytes * NumberChainsToRegen * sizeof(unsigned char));
    memset(HOST_Interleaved_Chains_To_Regen, 0, this->HashLengthBytes * NumberChainsToRegen * sizeof(unsigned char));

    hashPasswordData chainData;
    // Password in our space to regen
    for (i = 0; i < NumberChainsToRegen; i++) {
        UINT4 base_offset;
        // Get the chain being interleaved
        //printf("Adding chain %d\n", (i + WU->StartPoint));
        chainData = this->ChainsToRegen->at(i + WU->StartPoint);
        for (j = 0; j < (this->HashLengthBytes / 4); j++) {
            base_offset = 4 * j * NumberChainsToRegen;
            base_offset += i * 4;
            for (k = 0; k < 4; k++) {
                HOST_Interleaved_Chains_To_Regen[base_offset + k] = chainData.password[j*4 + k]; 
            }
        }
    }

    cudaMemset((void *)DEVICE_Chains_To_Regen, 0, this->HashLengthBytes * NumberChainsToRegen * sizeof(unsigned char));
    cudaMemcpy(DEVICE_Chains_To_Regen, HOST_Interleaved_Chains_To_Regen, this->HashLengthBytes * NumberChainsToRegen * sizeof(unsigned char), cudaMemcpyHostToDevice);


    // Kernel time!
    // Number of chains completed
    ChainsCompleted = 0;
    // Where we are in the current chain
    CurrentChainStartOffset = 0;
    // Calculated on the host for modulus reasons
    CharsetOffset = 0;
    PasswordSpaceOffset = 0;
    StepsPerInvocation = 1000;

    // If kernel time is set to zero, run full chains.
    if (!this->ThreadData[data->threadID].kernelTimeMs) {
        StepsPerInvocation = this->TableHeader->getChainLength();
    }
    // While we haven't finished all the chains:
    while (ChainsCompleted < NumberChainsToRegen) {
        CurrentChainStartOffset = 0;

        while (CurrentChainStartOffset < this->TableHeader->getChainLength()) {
            // Calculate the right charset offset
            CharsetOffset = CurrentChainStartOffset % this->hostConstantCharsetLengths[0];
            // PasswordSpaceOffset: The offset into the password space we are using.
            // 0, 1, 2, etc.
            PasswordSpaceOffset = (ChainsCompleted / 
                    (this->ThreadData[data->threadID].CUDABlocks * this->ThreadData[data->threadID].CUDAThreads));

            kernelTimer.start();

            // Don't overrun the end of the chain
            if ((CurrentChainStartOffset + StepsPerInvocation) > this->TableHeader->getChainLength()) {
                StepsPerInvocation = this->TableHeader->getChainLength() - CurrentChainStartOffset;
            }

            this->Launch_CUDA_Kernel(DEVICE_Chains_To_Regen, this->DEVICE_Passwords[data->threadID],
                this->DEVICE_Hashes[data->threadID], PasswordSpaceOffset, CurrentChainStartOffset,
                StepsPerInvocation, CharsetOffset, this->DEVICE_Success[data->threadID], data);

            cudaThreadSynchronize();
            // Copy the success and password data to the host
            cudaMemcpy(this->HOST_Success[data->threadID], this->DEVICE_Success[data->threadID],
                this->NumberOfHashes * sizeof(unsigned char), cudaMemcpyDeviceToHost);
            cudaMemcpy(this->HOST_Passwords[data->threadID], this->DEVICE_Passwords[data->threadID],
                this->NumberOfHashes * MAX_PASSWORD_LENGTH * sizeof(unsigned char), cudaMemcpyDeviceToHost);

            // Do something with the passwords...
            this->outputFoundHashes(data);

            // If all hashes are found, no point in continuing.
            if (this->HashFile->GetUncrackedHashCount() == 0) {
                return;
            }

            float ref_time = kernelTimer.getElapsedTimeInMilliSec();

            ActiveThreads = (this->ThreadData[data->threadID].CUDABlocks * this->ThreadData[data->threadID].CUDAThreads);
            
            if ((NumberChainsToRegen - ChainsCompleted) < ActiveThreads) {
                ActiveThreads = (NumberChainsToRegen - ChainsCompleted);
            }

            if (!silent) {
                if (this->Display) {
                    this->Display->setThreadFractionDone(data->threadID,
                            (float)(((float)((float)ChainsCompleted * (float)this->TableHeader->getChainLength() +
                        (float)CurrentChainStartOffset * (float)ActiveThreads) /
                        (float)((float)NumberChainsToRegen * (float)this->TableHeader->getChainLength()))));
                    this->Display->setThreadCrackSpeed(data->threadID, GPU_THREAD, ((ActiveThreads * StepsPerInvocation) / 1000) / ref_time);
                } else {
                    printf("Kernel Time: %0.3f ms  Step rate: %0.2f M/s Done: %0.2f%%    \r",ref_time,
                        ((ActiveThreads * StepsPerInvocation) / 1000) / ref_time,
                        (float)(100 * ((float)((float)ChainsCompleted * (float)this->TableHeader->getChainLength() +
                        (float)CurrentChainStartOffset * (float)ActiveThreads) /
                        (float)((float)NumberChainsToRegen * (float)this->TableHeader->getChainLength()))));
                    fflush(stdout);
                }
            }

            CurrentChainStartOffset += StepsPerInvocation;
            
            if (this->ThreadData[data->threadID].kernelTimeMs) {
                // Adjust the steps per invocation if needed.
                if ((ref_time > 1.1 * (float)this->ThreadData[data->threadID].kernelTimeMs) || (ref_time < 0.9 * (float)this->ThreadData[data->threadID].kernelTimeMs)) {
                    StepsPerInvocation = (UINT4)((float)StepsPerInvocation * ((float)this->ThreadData[data->threadID].kernelTimeMs / ref_time));
                    //printf("Adjusted SPI to %d\n", StepsPerInvocation);
                }
            }

        }
        ChainsCompleted += (this->ThreadData[data->threadID].CUDABlocks * this->ThreadData[data->threadID].CUDAThreads);
    }
    

    //printf("Freeing chains to regen.\n");
    // Free the chains we were working on.
    cudaFree(DEVICE_Chains_To_Regen);
    free(HOST_Interleaved_Chains_To_Regen);
}


void GRTRegenerateChains::setCommandLineData(GRTCrackCommandLineData *NewCommandLineData) {
    this->CommandLineData = NewCommandLineData;
}

void GRTRegenerateChains::setTableHeader(GRTTableHeader *NewTableHeader) {
    this->TableHeader = NewTableHeader;
}
void GRTRegenerateChains::setWorkunit(GRTWorkunit *NewWorkunit) {
    this->Workunit = NewWorkunit;
}
void GRTRegenerateChains::setHashfile(GRTHashFilePlain* NewHashfile) {
    this->HashFile = NewHashfile;
}
void GRTRegenerateChains::setDisplay(GRTCrackDisplay *NewDisplay) {
    this->Display = NewDisplay;
}



void GRTRegenerateChains::AllocatePerGPUMemory(GRTRegenerateThreadRunData *data) {
    // Malloc space on the GPU for everything.
    cudaError_t err;
    
    // Default flags are for memory on device and copying things.
    unsigned int flags = 0;

    // If we are using zero copy, set the zero copy flag.
    if (this->CommandLineData->GetUseZeroCopy()) {
        flags = cudaHostAllocMapped;
    }
    // Malloc device hash space.
    CH_CUDA_SAFE_CALL(cudaMalloc((void **)&this->DEVICE_Hashes[data->threadID],
        this->NumberOfHashes * this->HashLengthBytes * sizeof(unsigned char)));
    CH_CUDA_SAFE_CALL(cudaMemcpy(this->DEVICE_Hashes[data->threadID], this->HashList,
        this->NumberOfHashes * this->HashLengthBytes * sizeof(unsigned char), cudaMemcpyHostToDevice));

    //this->HOST_Success[data->threadID] = new unsigned char [this->NumberOfHashes * sizeof(unsigned char)];
    cudaHostAlloc((void **)&this->HOST_Success[data->threadID],
        this->NumberOfHashes * sizeof(unsigned char), flags);
    memset(this->HOST_Success[data->threadID], 0, this->NumberOfHashes * sizeof(unsigned char));
    this->HOST_Success_Reported[data->threadID] = new unsigned char [this->NumberOfHashes * sizeof(unsigned char)];
    memset(this->HOST_Success_Reported[data->threadID], 0, this->NumberOfHashes * sizeof(unsigned char));

    // If zero copy is in use, get the device pointer
    if (this->CommandLineData->GetUseZeroCopy()) {
        cudaHostGetDevicePointer((void **)&this->DEVICE_Success[data->threadID],
            this->HOST_Success[data->threadID], 0);
    } else {
        CH_CUDA_SAFE_CALL(cudaMalloc((void **)&this->DEVICE_Success[data->threadID],
            this->NumberOfHashes * sizeof(unsigned char)));
        CH_CUDA_SAFE_CALL(cudaMemset(this->DEVICE_Success[data->threadID], 0,
            this->NumberOfHashes * sizeof(unsigned char)));
    }

    //this->HOST_Passwords[data->threadID] = new unsigned char[MAX_PASSWORD_LEN * this->NumberOfHashes * sizeof(unsigned char)];
    cudaHostAlloc((void **)&this->HOST_Passwords[data->threadID],
        MAX_PASSWORD_LENGTH * this->NumberOfHashes * sizeof(unsigned char), flags);
    memset(this->HOST_Passwords[data->threadID], 0, MAX_PASSWORD_LENGTH * this->NumberOfHashes * sizeof(unsigned char));

    if (this->CommandLineData->GetUseZeroCopy()) {
        cudaHostGetDevicePointer((void **)&this->DEVICE_Passwords[data->threadID],
            this->HOST_Passwords[data->threadID], 0);
    } else {
        CH_CUDA_SAFE_CALL(cudaMalloc((void **)&this->DEVICE_Passwords[data->threadID],
            MAX_PASSWORD_LENGTH * this->NumberOfHashes * sizeof(unsigned char)));
        CH_CUDA_SAFE_CALL(cudaMemset(this->DEVICE_Passwords[data->threadID], 0,
            MAX_PASSWORD_LENGTH * this->NumberOfHashes * sizeof(unsigned char)));
    }

    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d: CUDA error 5: %s. Exiting.\n",
                data->threadID, cudaGetErrorString( err));
        return;
    }
    //printf("Memory for thread %d allocated.\n", data->threadID);
}

void GRTRegenerateChains::FreePerGPUMemory(GRTRegenerateThreadRunData *data) {
    CH_CUDA_SAFE_CALL(cudaFree(this->DEVICE_Hashes[data->threadID]));

    CH_CUDA_SAFE_CALL(cudaFreeHost(this->HOST_Success[data->threadID]));
    CH_CUDA_SAFE_CALL(cudaFreeHost(this->HOST_Passwords[data->threadID]));
    // Only free the device memory if zero copy was NOT used
    if (!this->CommandLineData->GetUseZeroCopy()) {
        CH_CUDA_SAFE_CALL(cudaFree(this->DEVICE_Passwords[data->threadID]));
        CH_CUDA_SAFE_CALL(cudaFree(this->DEVICE_Success[data->threadID]));
   }

    delete[] this->HOST_Success_Reported[data->threadID];
    //printf("Memory for thread %d freed.\n", data->threadID);
}

void GRTRegenerateChains::createConstantBitmap8kb() {
    // Initialize the 8kb shared memory bitmap.
    // 8192 bytes, 8 bits per byte.
    // Bits are determined by the first 16 bits of a hash.
    // Byte index is 13 bits, bit index is 3 bits.
    // Algorithm: To set:
    // First 13 bits of the hash (high order bits) are used as the
    // index to the array.
    // Next 3 bits control the left-shift amount of the '1' bit.
    // So, hash 0x0000 has byte 0, LSB set.
    // Hash 0x0105 has byte
    uint64_t i;
    unsigned char bitmap_byte;
    uint32_t bitmap_index;
    // Zero bitmap
    memset(this->hostConstantBitmap, 0, 8192);

    for (i = 0; i < this->NumberOfHashes; i++) {
        // Load first two bytes of hash - reversed order to work with the little endian memory storage.  Otherwise the GPU has to flip things around
        // and this is operations that should not have to be performed.
        bitmap_index = (this->HashList[i * this->HashLengthBytes + 1] << 8) + this->HashList[(i * this->HashLengthBytes)];
        //printf("Hash %lu: 2 bytes: %02X %02X ", i, this->ActiveHashList[i * 16], this->ActiveHashList[i * 16 + 1]);
        //printf("Bitmap index: %04X\n", bitmap_index);
        // Shift left by the lowest 3 bits
        bitmap_byte = 0x01 << (bitmap_index & 0x0007);
        bitmap_index = bitmap_index >> 3;
        //printf(" Index %u, byte %02X\n", bitmap_index, bitmap_byte);
        this->hostConstantBitmap[bitmap_index] |= bitmap_byte;
    }
    if (false) {
        printf("Bitmap output\n");
        for (i = 0; i < 8192; i++) {
            if (i % 4 == 0) {
                printf("\n Index %llu: ", i);
            }
            printf("%02X ", this->hostConstantBitmap[i]);
        }
    }
}


void GRTRegenerateChains::setChainsToRegen(std::vector<hashPasswordData>* newChainsToRegen) {
    this->ChainsToRegen = newChainsToRegen;
}


int GRTRegenerateChains::outputFoundHashes(struct GRTRegenerateThreadRunData *data) {
  int i, j;
  int totalPasswordsFound = 0;
  int passwordFound;
    for (i = 0; i < this->NumberOfHashes; i++) {
        if (this->HOST_Success[data->threadID][i] && !this->HOST_Success_Reported[data->threadID][i]) {

            // Store the return value - 0 if already reported, else 1.
            passwordFound = this->HashFile->ReportFoundPassword(&this->HashList[i * this->HashLengthBytes],
                    &this->HOST_Passwords[data->threadID][MAX_PASSWORD_LENGTH * i]);

            if (passwordFound) {
                if (this->Display) {
                    this->Display->addCrackedPassword((char *)&this->HOST_Passwords[data->threadID][MAX_PASSWORD_LENGTH * i]);
                } else {
                    this->HashFile->PrintNewFoundHashes();
                }
                // Only add one to password count if this is not a duplicate.
                totalPasswordsFound++;
            }
            this->HOST_Success_Reported[data->threadID][i] = 1;

            //printf("\nFound pass! %s\n", &this->HOST_Passwords[data->threadID][MAX_PASSWORD_LENGTH * i]);
            /*
            for (j = 0; j < this->passwordLength; j++) {
                this->statusBuffer[j] = this->HOST_Passwords[data->threadID][MAX_PASSWORD_LEN * i + j];
            }
            this->statusBuffer[j] = 0;
            this->Display->addCrackedPassword(this->statusBuffer);
            */
        }
    }
    if (this->Display) {
        this->Display->addCrackedHashes(totalPasswordsFound);
    }
    return totalPasswordsFound;
}
