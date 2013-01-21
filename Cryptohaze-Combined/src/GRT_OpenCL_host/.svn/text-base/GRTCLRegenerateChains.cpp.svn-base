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

#include "GRT_OpenCL_host/GRTCLRegenerateChains.h"
//#include <unistd.h>
#include <string.h>
#include "CH_Common/Timer.h"

extern char silent;

// Entry points for pthreads
extern "C" {
    void *GRTCLRegenChainGPUThread(void *);
}

void *GRTCLRegenChainGPUThread(void * pointer) {
    struct GRTCLRegenerateThreadRunData *data;

    data = (GRTCLRegenerateThreadRunData *) pointer;

    data->RegenerateChains->GPU_Thread(pointer);
    fflush(stdout);
#if USE_BOOST_THREADS
    return NULL;
#else
    pthread_exit(NULL);
#endif
}


GRTCLRegenerateChains::GRTCLRegenerateChains(int hashLengthBytes) {
    this->HashLengthBytes = hashLengthBytes;
    // Initialize Multithreading data to all null
    this->ActiveThreadCount = 0;
    memset(this->ThreadData, 0, MAX_SUPPORTED_THREADS * sizeof(GRTCLRegenerateThreadRunData));
#if USE_BOOST_THREADS
    memset(this->ThreadObjects, 0, MAX_SUPPORTED_THREADS * sizeof(boost::thread *));
#else
    memset(this->ThreadIds, 0, MAX_SUPPORTED_THREADS * sizeof(pthread_t));
#endif
    memset(this->OpenCLContexts, 0, MAX_SUPPORTED_THREADS * sizeof(CryptohazeOpenCL *));
    memset(this->DEVICE_Charset, 0, MAX_SUPPORTED_THREADS * sizeof(cl_mem));
    memset(this->DEVICE_Hashes, 0, MAX_SUPPORTED_THREADS * sizeof(cl_mem));
    memset(this->DEVICE_Passwords, 0, MAX_SUPPORTED_THREADS * sizeof(cl_mem));
    memset(this->DEVICE_Success, 0, MAX_SUPPORTED_THREADS * sizeof(cl_mem));
}


// Add a GPU deviceID to the list of active devices.
// Returns 0 on failure (probably too many threads), 1 on success.
int GRTCLRegenerateChains::addGPUDeviceID(int deviceId) {

    // Ensure the CommandLineData class has been added.
    // This is likely a programmer error, not a runtime error.
    // Therefore, exit.
    if (!this->CommandLineData) {
        printf("GRTCLRegenerateChains: Must add command line data first!\n");
        exit(1);
    }
/*
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
*/
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
    this->ThreadData[this->ActiveThreadCount].OpenCLDeviceId = deviceId;
    this->ThreadData[this->ActiveThreadCount].OpenCLPlatformId =
            this->CommandLineData->getOpenCLPlatform();
    this->ThreadData[this->ActiveThreadCount].threadID =
            this->ActiveThreadCount;
    this->ThreadData[this->ActiveThreadCount].RegenerateChains = this;


    // Set up the device parameters
    // If they are manually forced, set them, else set them to defaults.
    // CUDA Blocks
    if (this->CommandLineData->getWorkgroups()) {
        this->ThreadData[this->ActiveThreadCount].OpenCLWorkgroups =
            this->CommandLineData->getWorkgroups();
    } else {
        this->ThreadData[this->ActiveThreadCount].OpenCLWorkgroups = 0;
    }

    // CUDA Threads
    if (this->CommandLineData->getWorkitems()) {
        this->ThreadData[this->ActiveThreadCount].OpenCLWorkitems =
            this->CommandLineData->getWorkitems();
    } else {
        this->ThreadData[this->ActiveThreadCount].OpenCLWorkitems = 0;
    }

    // Default execution time
    if (this->CommandLineData->getKernelTimeMs()) {
        this->ThreadData[this->ActiveThreadCount].kernelTimeMs =
            this->CommandLineData->getKernelTimeMs();
    } else {
        this->ThreadData[this->ActiveThreadCount].kernelTimeMs =
                DEFAULT_CUDA_EXECUTION_TIME;
    }

    // Increment the active thread count.
    this->ActiveThreadCount++;

    //printf("Added GPU device %d\n", deviceId);

    return 1;
}


int GRTCLRegenerateChains::regenerateChains() {
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
        this->ThreadObjects[i] = new boost::thread(&GRTCLRegenChainGPUThread, &this->ThreadData[i]);
#else
        pthread_create(&this->ThreadIds[i], NULL, GRTCLRegenChainGPUThread, &this->ThreadData[i] );
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

    return 1;
}



// This is the GPU thread where we do the per-GPU tasks.
void GRTCLRegenerateChains::GPU_Thread(void *pointer) {
    struct GRTCLRegenerateThreadRunData *data;
    GRTWorkunitElement *WU;
    char buildOptions[1024];
    int vectorWidth = 1;

    data = (GRTCLRegenerateThreadRunData *) pointer;

    // Set up the OpenCL context for this thread.
    this->OpenCLContexts[data->threadID] = new CryptohazeOpenCL();

    this->OpenCLContexts[data->threadID]->selectPlatformById(data->OpenCLPlatformId);
    this->OpenCLContexts[data->threadID]->selectDeviceById(data->OpenCLDeviceId);
    this->OpenCLContexts[data->threadID]->createContext();

    this->OpenCLContexts[data->threadID]->createCommandQueue();
    this->OpenCLCommandQueue[data->threadID] = this->OpenCLContexts[data->threadID]->getCommandQueue();

    // Sort out workitem/workgroup count
    if (this->ThreadData[data->threadID].OpenCLWorkitems == 0) {
        this->ThreadData[data->threadID].OpenCLWorkitems = 
                this->OpenCLContexts[data->threadID]->getDefaultThreadCount();
    }
    if (this->ThreadData[data->threadID].OpenCLWorkgroups == 0) {
        this->ThreadData[data->threadID].OpenCLWorkgroups = 
                this->OpenCLContexts[data->threadID]->getDefaultBlockCount() * 
                this->OpenCLContexts[data->threadID]->getDefaultThreadCount();
    }


    // Handle generating the kernel/etc...
    sprintf(buildOptions, "-D PASSWORD_LENGTH=%d", this->TableHeader->getPasswordLength());

    // Try for the kernel
    if (this->CommandLineData->getUseAmdKernels()) {
        vectorWidth = this->CommandLineData->getVectorWidth();
        sprintf(buildOptions, "%s -D VECTOR_WIDTH=%d", buildOptions, vectorWidth);
        sprintf(buildOptions, "%s -D BITALIGN", buildOptions);
        this->OpenCLContexts[data->threadID]->buildProgramFromManySourcesConcat(this->getHashFileName(), buildOptions, this->getKernelSourceString());
        this->OpenCLContexts[data->threadID]->doAMDBFIPatch();
    } else {
        vectorWidth = this->CommandLineData->getVectorWidth();
        sprintf(buildOptions, "%s -D VECTOR_WIDTH=%d", buildOptions, vectorWidth);
        this->OpenCLContexts[data->threadID]->buildProgramFromManySourcesConcat(this->getHashFileName(), buildOptions, this->getKernelSourceString());
    }


    this->AllocatePerGPUMemory(data);
    this->copyDataToConstant(data);

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
}




void GRTCLRegenerateChains::RunGPUWorkunit(GRTWorkunitElement *WU, GRTCLRegenerateThreadRunData *data) {
    //printf("GRTCLRegenerateChains::RunGPUWorkunit thread %d\n", data->threadID);
    //printf("Got startPoint %d\n", WU->StartPoint);
    //printf("Got endpoint %d\n", WU->EndPoint);
    //printf("Chains to regen: %d\n", WU->EndPoint - WU->StartPoint + 1);


    unsigned char *HOST_Interleaved_Chains_To_Regen;
    cl_mem DEVICE_Interleaved_Chains_To_Regen;
    UINT4 i, j, k;
    cl_uint ChainsCompleted, CurrentChainStartOffset, PasswordSpaceOffset, CharsetOffset, StepsPerInvocation;
    int ActiveThreads;

    // Calculate the number of chains being regen'd by this thread.
    // Fenceposting issues - add one to the number to regen.
    cl_uint NumberChainsToRegen = WU->EndPoint - WU->StartPoint + 1;

    cl_program RegenerateChainsProgram;
    cl_kernel RegenerateChainsKernel;
    cl_int errorCode;
    cl_event kernelLaunchEvent;

    cl_uint deviceChainLength;
    cl_uint deviceTableIndex;
    size_t numberWorkgroups;
    size_t numberWorkitems;
    
    // We want the TOTAL number of hashes, not just the uncracked ones!
    //cl_uint numberOfHashes = this->HashFile->GetTotalHashCount();
    // I do NOT know why I made the previous comment.
    // This is now copying the actual number of hashes on the device...
    cl_uint numberOfHashes = this->NumberOfHashes;
    
    char *HOST_Charset_Lengths;
    cl_uint HOST_Charset_Length;
    cl_uint Number_Of_Threads;

    Timer TotalExecutionTimer, InvocationTimer;

    numberWorkgroups = this->ThreadData[data->threadID].OpenCLWorkgroups;
    numberWorkitems = this->ThreadData[data->threadID].OpenCLWorkitems;

    RegenerateChainsProgram = this->OpenCLContexts[data->threadID]->getProgram();
    RegenerateChainsKernel = clCreateKernel (RegenerateChainsProgram, this->getHashKernelName().c_str(), &errorCode);

    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }


    deviceChainLength = this->TableHeader->getChainLength();
    deviceTableIndex = this->TableHeader->getTableIndex();
    HOST_Charset_Lengths = this->TableHeader->getCharsetLengths();
    HOST_Charset_Length = HOST_Charset_Lengths[0];
    Number_Of_Threads = numberWorkgroups;

    DEVICE_Interleaved_Chains_To_Regen =
        clCreateBuffer (this->OpenCLContexts[data->threadID]->getContext(),
            CL_MEM_READ_ONLY,
            MAX_PASSWORD_LENGTH * NumberChainsToRegen * sizeof(unsigned char),
            NULL,
            &errorCode);
    if (errorCode != CL_SUCCESS) {
        printf("ERROR: Cannot allocate GPU memory.\n");
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    errorCode = 0;
    errorCode |= clSetKernelArg (RegenerateChainsKernel, 0, sizeof(cl_mem), &this->DEVICE_Charset[data->threadID]);
    errorCode |= clSetKernelArg (RegenerateChainsKernel, 1, sizeof(cl_uint), &HOST_Charset_Length);
    errorCode |= clSetKernelArg (RegenerateChainsKernel, 2, sizeof(cl_uint), &deviceChainLength);
    errorCode |= clSetKernelArg (RegenerateChainsKernel, 3, sizeof(cl_uint), &deviceTableIndex);
    errorCode |= clSetKernelArg (RegenerateChainsKernel, 4, sizeof(cl_uint), &Number_Of_Threads);
    errorCode |= clSetKernelArg (RegenerateChainsKernel, 5, sizeof(cl_uint), &NumberChainsToRegen);
    errorCode |= clSetKernelArg (RegenerateChainsKernel, 6, sizeof(cl_uint), &numberOfHashes);
    errorCode |= clSetKernelArg (RegenerateChainsKernel, 7, sizeof(cl_mem), &this->DEVICE_Bitmap[data->threadID]);
    errorCode |= clSetKernelArg (RegenerateChainsKernel, 8, sizeof(cl_mem), &DEVICE_Interleaved_Chains_To_Regen);
    errorCode |= clSetKernelArg (RegenerateChainsKernel, 9, sizeof(cl_mem), &this->DEVICE_Passwords[data->threadID]);
    errorCode |= clSetKernelArg (RegenerateChainsKernel, 10, sizeof(cl_mem), &this->DEVICE_Hashes[data->threadID]);
    errorCode |= clSetKernelArg (RegenerateChainsKernel, 11, sizeof(cl_mem), &this->DEVICE_Success[data->threadID]);

    if (errorCode != CL_SUCCESS) {
        printf("clSetKernelArg Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }


    // Interleave chains for better GPU performance and coalescing.
    HOST_Interleaved_Chains_To_Regen = (unsigned char *)malloc(MAX_PASSWORD_LENGTH * NumberChainsToRegen * sizeof(unsigned char));

    memset(HOST_Interleaved_Chains_To_Regen, 0, MAX_PASSWORD_LENGTH * NumberChainsToRegen * sizeof(unsigned char));


    hashPasswordData chainData;
    // Password in our space to regen
    for (i = 0; i < NumberChainsToRegen; i++) {
        UINT4 base_offset;
        // Get the chain being interleaved
        //printf("Loading chain %d\n", i + WU->StartPoint);
        chainData = this->ChainsToRegen->at(i + WU->StartPoint);
        //printf("ChainData %d: %s\n", i, chainData.password);
        for (j = 0; j < (MAX_PASSWORD_LENGTH / 4); j++) {
            base_offset = 4 * j * NumberChainsToRegen;
            base_offset += i * 4;
            for (k = 0; k < 4; k++) {
                HOST_Interleaved_Chains_To_Regen[base_offset + k] = chainData.password[j*4 + k];
            }
        }
    }


    errorCode = clEnqueueWriteBuffer (this->OpenCLCommandQueue[data->threadID],
        DEVICE_Interleaved_Chains_To_Regen,
        CL_TRUE /* blocking write */,
        0 /* offset */,
        MAX_PASSWORD_LENGTH * NumberChainsToRegen * sizeof(unsigned char) /* bytes to copy */,
        HOST_Interleaved_Chains_To_Regen,
        NULL, NULL, NULL /* event list stuff */);
    if (errorCode != CL_SUCCESS) {
        printf("clEnqueueWriteBuffer 2: Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }


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
    while (ChainsCompleted < (NumberChainsToRegen / this->CommandLineData->getVectorWidth())) {
        CurrentChainStartOffset = 0;

        while (CurrentChainStartOffset < this->TableHeader->getChainLength()) {
            // Calculate the right charset offset
            CharsetOffset = CurrentChainStartOffset % this->hostConstantCharsetLengths[0];
            // PasswordSpaceOffset: The offset into the password space we are using.
            // 0, 1, 2, etc.
            PasswordSpaceOffset = (ChainsCompleted /
                    this->ThreadData[data->threadID].OpenCLWorkgroups);

            // Don't overrun the end of the chain
            if ((CurrentChainStartOffset + StepsPerInvocation) > this->TableHeader->getChainLength()) {
                StepsPerInvocation = this->TableHeader->getChainLength() - CurrentChainStartOffset;
            }

            errorCode = 0;
            errorCode |= clSetKernelArg (RegenerateChainsKernel, 12, sizeof(cl_uint), &PasswordSpaceOffset);
            errorCode |= clSetKernelArg (RegenerateChainsKernel, 13, sizeof(cl_uint), &CurrentChainStartOffset);
            errorCode |= clSetKernelArg (RegenerateChainsKernel, 14, sizeof(cl_uint), &StepsPerInvocation);

            if (errorCode != CL_SUCCESS) {
                printf("Error: %s\n", print_cl_errstring(errorCode));
                exit(1);
            }

            InvocationTimer.start();
            // Actually launch the kernel!
            errorCode = clEnqueueNDRangeKernel(this->OpenCLCommandQueue[data->threadID],
                    RegenerateChainsKernel,
                    1 /* numDims */,
                    NULL /* offset */,
                    &numberWorkgroups,
                    &numberWorkitems,
                    NULL, NULL,
                    &kernelLaunchEvent);


            if (clWaitForEvents(1, &kernelLaunchEvent) != CL_SUCCESS) {
                printf("\nError on wait for event!\n");
                fflush(stdout);
            };

            errorCode = clEnqueueReadBuffer (this->OpenCLCommandQueue[data->threadID],
                this->DEVICE_Success[data->threadID],
                CL_TRUE /* blocking write */,
                0 /* offset */,
                this->NumberOfHashes * sizeof(unsigned char) /* bytes to copy */,
                this->HOST_Success[data->threadID],
                NULL, NULL, NULL /* event list stuff */);
            errorCode |= clEnqueueReadBuffer (this->OpenCLCommandQueue[data->threadID],
                this->DEVICE_Passwords[data->threadID],
                CL_TRUE /* blocking write */,
                0 /* offset */,
                this->NumberOfHashes * MAX_PASSWORD_LENGTH * sizeof(unsigned char) /* bytes to copy */,
                this->HOST_Passwords[data->threadID],
                NULL, NULL, NULL /* event list stuff */);
            if (errorCode != CL_SUCCESS) {
                printf("Error: %s\n", print_cl_errstring(errorCode));
                exit(1);
            }

            // Do something with the passwords...
            this->outputFoundHashes(data);

            // If all hashes are found, no point in continuing.
            if (this->HashFile->GetUncrackedHashCount() == 0) {
                return;
            }

            float ref_time = InvocationTimer.stop() * 1000.0;

			// Correct for timer issues with short time periods on Windows.
			if (ref_time < 10) {
				ref_time = 10.0;
			}
            
            ActiveThreads = (this->ThreadData[data->threadID].OpenCLWorkgroups * this->CommandLineData->getVectorWidth());

            if ((NumberChainsToRegen - ChainsCompleted) < ActiveThreads) {
                ActiveThreads = (NumberChainsToRegen - ChainsCompleted);
            }

            if (!silent) {
                if (this->Display) {
                    this->Display->setThreadFractionDone(data->threadID,
                            (float)(((float)((float)ChainsCompleted * this->CommandLineData->getVectorWidth() * (float)this->TableHeader->getChainLength() +
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
					if (StepsPerInvocation < 100) {
						StepsPerInvocation = 100;
					}
                }
            }

        }
        ChainsCompleted += this->ThreadData[data->threadID].OpenCLWorkgroups;
    }

    clReleaseMemObject(DEVICE_Interleaved_Chains_To_Regen);
    free(HOST_Interleaved_Chains_To_Regen);
}


void GRTCLRegenerateChains::setCommandLineData(GRTCLCrackCommandLineData *NewCommandLineData) {
    this->CommandLineData = NewCommandLineData;
}

void GRTCLRegenerateChains::setTableHeader(GRTTableHeader *NewTableHeader) {
    this->TableHeader = NewTableHeader;
}
void GRTCLRegenerateChains::setWorkunit(GRTWorkunit *NewWorkunit) {
    this->Workunit = NewWorkunit;
}
void GRTCLRegenerateChains::setHashfile(GRTHashFilePlain* NewHashfile) {
    this->HashFile = NewHashfile;
}
void GRTCLRegenerateChains::setDisplay(GRTCrackDisplay *NewDisplay) {
    this->Display = NewDisplay;
}



void GRTCLRegenerateChains::AllocatePerGPUMemory(GRTCLRegenerateThreadRunData *data) {
    // Malloc space on the GPU for everything.
    cl_int errorCode;

    this->DEVICE_Hashes[data->threadID] =
            clCreateBuffer (this->OpenCLContexts[data->threadID]->getContext(),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            this->NumberOfHashes * this->HashLengthBytes * sizeof(unsigned char),
            this->HashList,
            &errorCode);
    if (errorCode != CL_SUCCESS) {
        printf("ERROR: Cannot allocate GPU memory.\n");
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    this->DEVICE_Charset[data->threadID] = clCreateBuffer (this->OpenCLContexts[data->threadID]->getContext(),
        CL_MEM_READ_WRITE,
        512 * sizeof(unsigned char),
        NULL,
        &errorCode);
    if (errorCode != CL_SUCCESS) {
        printf("ERROR: Cannot allocate GPU memory.  Try a smaller number of chains.\n");
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    this->HOST_Success[data->threadID] = (unsigned char *)malloc(this->NumberOfHashes * sizeof(unsigned char));
    memset(this->HOST_Success[data->threadID], 0, this->NumberOfHashes * sizeof(unsigned char));
    this->HOST_Success_Reported[data->threadID] = (unsigned char *)malloc(this->NumberOfHashes * sizeof(unsigned char));
    memset(this->HOST_Success_Reported[data->threadID], 0, this->NumberOfHashes * sizeof(unsigned char));

    this->DEVICE_Success[data->threadID] =
        clCreateBuffer (this->OpenCLContexts[data->threadID]->getContext(),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            this->NumberOfHashes * sizeof(unsigned char),
            this->HOST_Success[data->threadID],
            &errorCode);
    if (errorCode != CL_SUCCESS) {
        printf("ERROR: Cannot allocate GPU memory.\n");
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    this->HOST_Passwords[data->threadID] = (unsigned char *)malloc(MAX_PASSWORD_LENGTH * this->NumberOfHashes * sizeof(unsigned char));
    memset(this->HOST_Passwords[data->threadID], 0, MAX_PASSWORD_LENGTH * this->NumberOfHashes * sizeof(unsigned char));

    this->DEVICE_Passwords[data->threadID] =
        clCreateBuffer (this->OpenCLContexts[data->threadID]->getContext(),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            MAX_PASSWORD_LENGTH * this->NumberOfHashes * sizeof(unsigned char),
            this->HOST_Passwords[data->threadID],
            &errorCode);
    if (errorCode != CL_SUCCESS) {
        printf("ERROR: Cannot allocate GPU memory.\n");
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    this->DEVICE_Bitmap[data->threadID] =
        clCreateBuffer (this->OpenCLContexts[data->threadID]->getContext(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            8192,
            this->hostConstantBitmap,
            &errorCode);
    if (errorCode != CL_SUCCESS) {
        printf("ERROR: Cannot allocate GPU memory.\n");
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    //printf("Memory for thread %d allocated.\n", data->threadID);
}

void GRTCLRegenerateChains::FreePerGPUMemory(GRTCLRegenerateThreadRunData *data) {
/*    CUDA_SAFE_CALL(cudaFree(this->DEVICE_Hashes[data->threadID]));

    CUDA_SAFE_CALL(cudaFreeHost(this->HOST_Success[data->threadID]));
    CUDA_SAFE_CALL(cudaFreeHost(this->HOST_Passwords[data->threadID]));
    // Only free the device memory if zero copy was NOT used
    if (!this->CommandLineData->GetUseZeroCopy()) {
        CUDA_SAFE_CALL(cudaFree(this->DEVICE_Passwords[data->threadID]));
        CUDA_SAFE_CALL(cudaFree(this->DEVICE_Success[data->threadID]));
   }

    delete[] this->HOST_Success_Reported[data->threadID];
    //printf("Memory for thread %d freed.\n", data->threadID);
 */
}

void GRTCLRegenerateChains::createConstantBitmap8kb() {
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


void GRTCLRegenerateChains::setChainsToRegen(std::vector<hashPasswordData>* newChainsToRegen) {
    this->ChainsToRegen = newChainsToRegen;
}


int GRTCLRegenerateChains::outputFoundHashes(struct GRTCLRegenerateThreadRunData *data) {
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


// Copy all the various data into the GPU with the needed transfer function
void GRTCLRegenerateChains::copyDataToConstant(GRTCLRegenerateThreadRunData *data) {
    char hostCharset[512]; // The 512 byte array copied to the GPU
    int i;
    char** hostCharset2D; // The 16x256 array of characters
    uint32_t charsetLength;
    char *CharsetLengths;
    uint32_t numberThreads;
    cl_int errorCode;

    hostCharset2D = this->TableHeader->getCharset();
    CharsetLengths = this->TableHeader->getCharsetLengths();
    numberThreads = this->ThreadData[data->threadID].OpenCLWorkgroups;

    charsetLength = CharsetLengths[0];

    //printf("Charset length: %d\n", charsetLength);
    //printf("\n\nNUMBER THREAD IN COPY CONSTANT: %d\n", numberThreads);

    for (i = 0; i < 512; i++) {
        hostCharset[i] = hostCharset2D[0][i % charsetLength];
    }

    errorCode = clEnqueueWriteBuffer (this->OpenCLCommandQueue[data->threadID],
        this->DEVICE_Charset[data->threadID],
        CL_TRUE /* blocking write */,
        0 /* offset */,
        512 /* bytes to copy */,
        (void *)hostCharset,
        NULL, NULL, NULL /* event list stuff */);
    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    for (i = 0; i < 16; i++) {
        delete[] hostCharset2D[i];
    }
    delete[] hostCharset2D;

    delete[] CharsetLengths;

    return;
}