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

#include "GRT_OpenCL_host/GRTCLCandidateHashes.h"
#include "CH_Common/GRTWorkunit.h"
#include <vector>
//#include <unistd.h>
#include <string.h>
#include "CH_Common/Timer.h"

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


GRTCLCandidateHashes::GRTCLCandidateHashes(int hashLengthBytes) {
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

    memset(this->OpenCLContexts, 0, MAX_SUPPORTED_THREADS * sizeof(CryptohazeOpenCL *));

}

int GRTCLCandidateHashes::addHashToCrack(hashPasswordData *hashToAdd, int hashLength) {
    this->hashesToCrack.push_back(*hashToAdd);
	return 1;
}

// Add a GPU deviceID to the list of active devices.
// Returns 0 on failure (probably too many threads), 1 on success.
int GRTCLCandidateHashes::addGPUDeviceID(int deviceId) {

    // Ensure the CommandLineData class has been added.
    // This is likely a programmer error, not a runtime error.
    // Therefore, exit.
    if (!this->CommandLineData) {
        printf("GRTCLCandidateHashes: Must add command line data first!\n");
        exit(1);
    }
/*
    // If the device ID is invalid, reject.
    // This is likely a runtime error, so just return 0.
    if (deviceId >= this->CommandLineData->getCudaNumberDevices()) {
        sprintf(this->statusStrings, "Invalid device ID %d", deviceId);
        this->Display->addStatusLine(this->statusStrings);
        return 0;
    }
*/
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
    this->ThreadData[this->ActiveThreadCount].OpenCLDeviceId = deviceId;
    this->ThreadData[this->ActiveThreadCount].OpenCLPlatformId =
            this->CommandLineData->getOpenCLPlatform();
    this->ThreadData[this->ActiveThreadCount].threadID =
            this->ActiveThreadCount;
    this->ThreadData[this->ActiveThreadCount].CandidateHashes = this;


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

    return 1;
}


int GRTCLCandidateHashes::generateCandidateHashes() {
    int i;

    // All the memory allocs/etc are per-thread.

    // All global work done.  Onto the per-thread work.
    //printf("Creating %d threads\n", this->ActiveThreadCount);
    // Enter all the threads
    for(i = 0; i < this->ActiveThreadCount; i++) {
#if USE_BOOST_THREADS
        this->ThreadObjects[i] = new boost::thread(&CHHashTypeGPUThread, &this->ThreadData[i]);
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
void GRTCLCandidateHashes::GPU_Thread(void *pointer) {
    struct GRTThreadRunData *data;
    GRTWorkunitElement *WU;
    char buildOptions[1024];
    int vectorWidth = 1;

    data = (GRTThreadRunData *) pointer;

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
    //printf("Using vector width %d\n!!!", this->CommandLineData->getVectorWidth());
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
    delete this->OpenCLContexts[data->threadID];
}

void GRTCLCandidateHashes::RunGPUWorkunit(GRTWorkunitElement *WU, GRTThreadRunData *data) {
    unsigned char *hash;
    unsigned char *HOST_Return_Hashes;
    UINT4 ChainsCompleted, ThreadSpaceOffset, ThreadSpaceOffsetMax, StartStep, StepsPerInvocation;
    unsigned int timer, totaltimer;
    //float TotalStepsNeeded;
    UINT4 base_offset, i, j, k;

    // For adding them into the list of generated hashes
    hashData candidateHashResult;

    cl_program CandidateHashProgram;
    cl_kernel CandidateHashKernel;
    cl_int errorCode;
    cl_event kernelLaunchEvent;

    cl_uint deviceChainLength;
    cl_uint deviceTableIndex;
    size_t numberWorkgroups;
    size_t numberWorkitems;

    char *HOST_Charset_Lengths;
    uint32_t HOST_Charset_Length;
    uint32_t Number_Of_Threads;

    Timer TotalExecutionTimer, InvocationTimer;

    //printf("GRTCLCandidateHashes::RunGPUWorkunit thread %d\n", data->threadID);


    numberWorkgroups = this->ThreadData[data->threadID].OpenCLWorkgroups;
    numberWorkitems = this->ThreadData[data->threadID].OpenCLWorkitems;

    

    CandidateHashProgram = this->OpenCLContexts[data->threadID]->getProgram();
    CandidateHashKernel = clCreateKernel (CandidateHashProgram, this->getHashKernelName().c_str(), &errorCode);

    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    //printf("Kernel built successfully!\n");



    // Alright.  We have a workunit, and we have a hash.  Let's rock!

    // Allocate memory for the hash we are about to crack
    hash = new unsigned char[this->HashLengthBytes];

    // Copy the appropriate hash into the GPU constant memory.
    for (i = 0; i < this->HashLengthBytes; i++) {
        hash[i] = this->hashesToCrack[WU->StartPoint].hash[i];
    }

    errorCode = clEnqueueWriteBuffer (this->OpenCLCommandQueue[data->threadID],
        this->DEVICE_Hash[data->threadID],
        CL_TRUE /* blocking write */,
        0 /* offset */,
        this->HashLengthBytes /* bytes to copy */,
        (void *)hash,
        NULL, NULL, NULL /* event list stuff */);
    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }


    deviceChainLength = this->TableHeader->getChainLength();
    deviceTableIndex = this->TableHeader->getTableIndex();
    HOST_Charset_Lengths = this->TableHeader->getCharsetLengths();
    HOST_Charset_Length = HOST_Charset_Lengths[0];
    Number_Of_Threads = numberWorkgroups;
    

    errorCode = 0;
    errorCode |= clSetKernelArg (CandidateHashKernel, 0, sizeof(cl_mem), &this->DEVICE_Charset[data->threadID]);
    errorCode |= clSetKernelArg (CandidateHashKernel, 1, sizeof(cl_uint), &HOST_Charset_Length);
    errorCode |= clSetKernelArg (CandidateHashKernel, 2, sizeof(cl_uint), &deviceChainLength);
    errorCode |= clSetKernelArg (CandidateHashKernel, 3, sizeof(cl_uint), &deviceTableIndex);
    errorCode |= clSetKernelArg (CandidateHashKernel, 4, sizeof(cl_uint), &Number_Of_Threads);
    errorCode |= clSetKernelArg (CandidateHashKernel, 5, sizeof(cl_mem), &this->DEVICE_Hash[data->threadID]);
    errorCode |= clSetKernelArg (CandidateHashKernel, 6, sizeof(cl_mem), &this->DEVICE_End_Hashes[data->threadID]);
    // The rest are iteration specific, so will have to wait.

    
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
    TotalExecutionTimer.start();
     // If kernel time is set to zero, run full chains (headless server).
    if (!this->ThreadData[data->threadID].kernelTimeMs) {
        StepsPerInvocation = this->TableHeader->getChainLength();
    }

    ThreadSpaceOffsetMax = (this->TableHeader->getChainLength() /
            (/*this->CommandLineData->getVectorWidth() * */this->ThreadData[data->threadID].OpenCLWorkgroups));
    ThreadSpaceOffsetMax += 1;

    //printf("ThreadSpaceOffsetMax: %d\n", ThreadSpaceOffsetMax);

    // We need to complete as many steps the chain length
    while (ChainsCompleted < this->TableHeader->getChainLength()) {
        // Offset in the space (run group)
        ThreadSpaceOffset = (ChainsCompleted /
                (this->ThreadData[data->threadID].OpenCLWorkgroups));
        if (!this->Display && !silent) {
            printf("\nRunning group %d of %d\n", ThreadSpaceOffset + 1,
                (this->TableHeader->getChainLength() /
                (this->ThreadData[data->threadID].OpenCLWorkgroups * this->CommandLineData->getVectorWidth())) + 1);
        }
        StartStep = 0;
        StepsPerInvocation = 100; // Reset this each time for sanity

        // Don't run past the end - if we will go past the end, work fewer steps.
        while (StartStep < this->TableHeader->getChainLength()) {
            if (StartStep + StepsPerInvocation > this->TableHeader->getChainLength()) {
                StepsPerInvocation = this->TableHeader->getChainLength() - StartStep;
            }

            errorCode = 0;
            errorCode |= clSetKernelArg (CandidateHashKernel, 7, sizeof(cl_uint), &ThreadSpaceOffset);
            errorCode |= clSetKernelArg (CandidateHashKernel, 8, sizeof(cl_uint), &StartStep);
            errorCode |= clSetKernelArg (CandidateHashKernel, 9, sizeof(cl_uint), &StepsPerInvocation);

            if (errorCode != CL_SUCCESS) {
                printf("Error: %s\n", print_cl_errstring(errorCode));
                exit(1);
            }

            InvocationTimer.start();
            // Actually launch the kernel!

            errorCode = clEnqueueNDRangeKernel(this->OpenCLCommandQueue[data->threadID],
                    CandidateHashKernel,
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

            /*
            this->runCandidateHashKernel(this->TableHeader->getPasswordLength(), this->ThreadData[data->threadID].CUDABlocks,
                this->ThreadData[data->threadID].CUDAThreads,
                this->DEVICE_End_Hashes[data->threadID], ThreadSpaceOffset, StartStep, StepsPerInvocation);
                */

            float ref_time = InvocationTimer.stop() * 1000;

			// Correct for timer issues with short time periods on Windows.
			if (ref_time < 10) {
				ref_time = 10.0;
			}

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

                //printf("\nKernel Time: %0.3f ms  Done: %0.2f%%         \n",ref_time,
                //    (float)(100.0 * (float)(StartStep) / (float)this->TableHeader->getChainLength()));
                //fflush(stdout);
            }
            StartStep += StepsPerInvocation;

            if (this->ThreadData[data->threadID].kernelTimeMs) {
                // Adjust the steps per invocation if needed.
                if ((ref_time > 1.1 * (float)this->ThreadData[data->threadID].kernelTimeMs) ||
                        (ref_time < 0.9 * (float)this->ThreadData[data->threadID].kernelTimeMs)) {
                    StepsPerInvocation = (UINT4)((float)StepsPerInvocation *
                            ((float)this->ThreadData[data->threadID].kernelTimeMs / ref_time));

                    if (StepsPerInvocation < 100) {
                        StepsPerInvocation = 100;
                    }
                }
            }
        }

        // Add the full block size to the number completed.
        ChainsCompleted += this->ThreadData[data->threadID].OpenCLWorkgroups * this->CommandLineData->getVectorWidth();
    }


    float total_time = TotalExecutionTimer.stop();
    if (this->Display) {
        this->Display->setThreadCrackSpeed(data->threadID, GPU_THREAD,
            (float)((float)this->TableHeader->getChainLength() * (float)this->TableHeader->getChainLength() / 2) / (total_time * 1000000.0));
    } else if (!silent) {
        printf("\n\n");
        printf("Total time: %0.2f seconds\n", total_time / 1000.0);
        printf("Average rate: %0.2f M h/s\n",
            (float)((float)this->TableHeader->getChainLength() * (float)this->TableHeader->getChainLength() / 2) / (total_time * 1000.0));
    }

    errorCode = clEnqueueReadBuffer (this->OpenCLCommandQueue[data->threadID],
        this->DEVICE_End_Hashes[data->threadID],
        CL_TRUE /* blocking write */,
        0 /* offset */,
        this->HashLengthBytes * this->TableHeader->getChainLength() * sizeof(unsigned char) /* bytes to copy */,
        this->HOST_End_Hashes[data->threadID],
        NULL, NULL, NULL /* event list stuff */);
    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    // Decoalesce the hashes so we have something sane
    // This can *probably* go straight into candidateHashResult...
    HOST_Return_Hashes = new unsigned char[this->HashLengthBytes * this->TableHeader->getChainLength()];

    for (i = 0; i < this->TableHeader->getChainLength(); i++) {
        //printf("Hash readback: Hash %d\n", i);
        fflush(stdout);
        // Segment
        for (j = 0; j < (this->HashLengthBytes / 4); j++) {
            base_offset = this->TableHeader->getChainLength() * j * 4; // Segment start
            base_offset += i * 4; // This chain start
            for (k = 0; k < 4; k++) {
                //printf("reading offset %d\n", base_offset + k);
                //printf("writing offset %d\n", i * this->HashLengthBytes + (j * 4) + k);
                HOST_Return_Hashes[i * this->HashLengthBytes + (j * 4) + k] = this->HOST_End_Hashes[data->threadID][base_offset + k];
            }
        }
    }
    //printf("Done with hash readback.\n");

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
std::vector<hashData> *GRTCLCandidateHashes::getGeneratedCandidates() {
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

void GRTCLCandidateHashes::setCommandLineData(GRTCLCrackCommandLineData *NewCommandLineData) {
    this->CommandLineData = NewCommandLineData;
}

void GRTCLCandidateHashes::setTableHeader(GRTTableHeader *NewTableHeader) {
    this->TableHeader = NewTableHeader;
}
void GRTCLCandidateHashes::setWorkunit(GRTWorkunit *NewWorkunit) {
    this->Workunit = NewWorkunit;
}

void GRTCLCandidateHashes::setDisplay(GRTCrackDisplay *NewDisplay) {
    this->Display = NewDisplay;
}



void GRTCLCandidateHashes::AllocatePerGPUMemory(GRTThreadRunData *data) {
    cl_int errorCode;
    
    // Allocate host memory for the end hashes.
    this->HOST_End_Hashes[data->threadID] =
            (unsigned char *)malloc(this->HashLengthBytes * this->TableHeader->getChainLength() * sizeof(unsigned char));


    this->DEVICE_End_Hashes[data->threadID] =
            clCreateBuffer (this->OpenCLContexts[data->threadID]->getContext(),
            CL_MEM_READ_WRITE,
            this->HashLengthBytes * this->TableHeader->getChainLength() * sizeof(unsigned char),
            NULL,
            &errorCode);
    if (errorCode != CL_SUCCESS) {
        printf("ERROR: Cannot allocate GPU memory.  Try a smaller number of chains.\n");
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    this->DEVICE_Charset[data->threadID] = clCreateBuffer (this->OpenCLContexts[data->threadID]->getContext(),
        CL_MEM_READ_ONLY,
        512 * sizeof(unsigned char),
        NULL,
        &errorCode);

    if (errorCode != CL_SUCCESS) {
        printf("ERROR: Cannot allocate GPU memory.  Try a smaller number of chains.\n");
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }


    this->DEVICE_Hash[data->threadID] = clCreateBuffer (this->OpenCLContexts[data->threadID]->getContext(),
        CL_MEM_READ_ONLY,
        this->HashLengthBytes * sizeof(unsigned char),
        NULL,
        &errorCode);

    if (errorCode != CL_SUCCESS) {
        printf("ERROR: Cannot allocate GPU memory.  Try a smaller number of chains.\n");
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }
}

void GRTCLCandidateHashes::FreePerGPUMemory(GRTThreadRunData *data) {
    //TODO: Fix why releasing memory crashes!
    clReleaseMemObject(this->DEVICE_End_Hashes[data->threadID]);
    clReleaseMemObject(this->DEVICE_Charset[data->threadID]);
    clReleaseMemObject(this->DEVICE_Hash[data->threadID]);
    free(this->HOST_End_Hashes[data->threadID]);
}

// Copy all the various data into the GPU with the needed transfer function
void GRTCLCandidateHashes::copyDataToConstant(GRTThreadRunData *data) {
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