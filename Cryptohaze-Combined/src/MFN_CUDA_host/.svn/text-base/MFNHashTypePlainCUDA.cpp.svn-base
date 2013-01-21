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


#include "CUDA_Common/CHCudaUtils.h"
#include "MFN_CUDA_host/MFNHashTypePlainCUDA.h"
#include "MFN_Common/MFNCommandLineData.h"
#include "MFN_Common/MFNMultiforcerClassFactory.h"
#include "CH_HashFiles/CHHashFileVPlain.h"
#include "MFN_Common/MFNDisplay.h"
#include "MFN_Common/MFNDebugging.h"
#include "MFN_Common/MFNNetworkClient.h"

#include "cuda.h"

extern MFNClassFactory MultiforcerGlobalClassFactory;


MFNHashTypePlainCUDA::MFNHashTypePlainCUDA(int hashLengthBytes) :  MFNHashTypePlain(hashLengthBytes) {
    trace_printf("MFNHashTypePlainCUDA::MFNHashTypePlainCUDA(%d)\n", hashLengthBytes);

    this->MFNHashTypeMutex.lock();
    this->threadId = MultiforcerGlobalClassFactory.getDisplayClass()->getFreeThreadId(GPU_THREAD);
    this->numberThreads++;
    trace_printf("MFNHashType GPU/CUDA Thread ID %d\n", this->threadId);
    this->MFNHashTypeMutex.unlock();

}

void MFNHashTypePlainCUDA::setupDevice() {
    trace_printf("CHHashTypeVPlainCUDA::setupDevice()\n");

    CHCUDAUtils *CudaUtils = MultiforcerGlobalClassFactory.getCudaUtilsClass();

    // Set the CUDA device
    trace_printf("Thread %d setting device to %d\n",this->threadId, this->gpuDeviceId);
    cudaSetDevice(this->gpuDeviceId);

    // If the user requests zerocopy and the device can handle it, add it.
    if (this->CommandLineData->GetUseZeroCopy() &&
            CudaUtils->getCudaCanMapHostMemory(this->gpuDeviceId)) {
        this->useZeroCopy = 1;
    }

    // If the device is integrated & can map memory, add it - integrated devices
    // are already sharing host memory, so no point in copying the data over.
    if (CudaUtils->getCudaIsIntegrated(this->gpuDeviceId) &&
            CudaUtils->getCudaCanMapHostMemory(this->gpuDeviceId)) {
        this->useZeroCopy = 1;
    }

    // Enable blocking sync.  This dramatically reduces CPU usage.
    // If zero copy is being used, set DeviceMapHost as well
    if (this->useZeroCopy) {
        cudaSetDeviceFlags(cudaDeviceBlockingSync | cudaDeviceMapHost);
    } else {
        cudaSetDeviceFlags(cudaDeviceBlockingSync);
    }
    
    // TODO: Override thread/block count setting by device shared memory size.

}


void MFNHashTypePlainCUDA::teardownDevice() {
    trace_printf("MFNHashTypePlainCUDA::teardownDevice()\n");
    //printf("Thread %d in teardownDevice.\n", this->threadId);
    // Free all thread resources to eliminate warnings with CUDA 4.0
    cudaThreadExit();
}

void MFNHashTypePlainCUDA::allocateThreadAndDeviceMemory() {
    trace_printf("MFNHashTypePlainCUDA::allocateThreadAndDeviceMemory()\n");
    /**
     * Error variable - stores the result of the various mallocs & such.
     */
    cudaError_t err, err2;

    /**
     * Flags for calling cudaHostMalloc - will be set to cudaHostAllocMapped
     * if we are mapping memory to the host with zero copy.
     */
    unsigned int cudaHostMallocFlags = 0;

    if (this->useZeroCopy) {
        cudaHostMallocFlags |= cudaHostAllocMapped;
    }


    /*
     * Malloc the device hashlist space.  This is the number of available hashes
     * times the hash length in bytes.  The data will be copied later.
     */
    err = cudaMalloc((void **)&this->DeviceHashlistAddress,
        this->activeHashesProcessed.size() * this->hashLengthBytes);
    if (err != cudaSuccess) {
        printf("Unable to allocate %d bytes for device hashlist!  Exiting!\n",
                this->activeHashesProcessed.size() * this->hashLengthBytes);
        printf("return code: %d\n", err);
        exit(1);
    }

    /*
     * Allocate the host/device space for the success list (flags for found passwords).
     * This is a byte per password.  To avoid atomic write issues, each password
     * gets a full addressible byte, and the GPU handles the dependencies between
     * multiple threads trying to set a flag in the same segment of memory.
     *
     * On the host, it will be allocated as mapped memory if we are using zerocopy.
     *
     * As this region of memory is frequently copied back to the host, mapping it
     * improves performance.  In theory.
     */
    err = cudaHostAlloc((void **)&this->HostSuccessAddress,
        this->activeHashesProcessed.size(), cudaHostMallocFlags);
    if (err != cudaSuccess) {
        printf("Unable to allocate %d bytes for success flags!  Exiting!\n",
                this->activeHashesProcessed.size());
        printf("return code: %d\n", err);
        exit(1);
    }

    // Clear host success flags region - if we are mapping the memory, the GPU
    // will directly write this.
    memset(this->HostSuccessAddress, 0, this->activeHashesProcessed.size());

    // Allocate memory for the reported flags.
    this->HostSuccessReportedAddress = new uint8_t [this->activeHashesProcessed.size()];
    memset(this->HostSuccessReportedAddress, 0, this->activeHashesProcessed.size());

    // If zero copy is in use, get the device pointer for the success data, else
    // malloc a region of memory on the device.
    if (this->useZeroCopy) {
        err = cudaHostGetDevicePointer((void **)&this->DeviceSuccessAddress,
            this->HostSuccessAddress, 0);
        err2 = cudaSuccess;
    } else {
        err = cudaMalloc((void **)&this->DeviceSuccessAddress,
            this->activeHashesProcessed.size());
        err2 = cudaMemset(this->DeviceSuccessAddress, 0,
            this->activeHashesProcessed.size());
    }
    if ((err != cudaSuccess) || (err2 != cudaSuccess)) {
        printf("Unable to allocate %d bytes for device success list!  Exiting!\n",
                this->activeHashesProcessed.size());
        printf("return code: %d\n", err);
        printf("return code: %d\n", err2);
        exit(1);
    }

    /*
     * Allocate memory for the found passwords.  As this is commonly copied
     * back and forth, it will be made zero copy if requested.
     *
     * This requires (number hashes * maxFoundPlainLength) bytes of data.
     */

    err = cudaHostAlloc((void **)&this->HostFoundPasswordsAddress,
        this->maxFoundPlainLength * this->activeHashesProcessed.size() , cudaHostMallocFlags);
    if (err != cudaSuccess) {
        printf("Unable to allocate %d bytes for host password list!  Exiting!\n",
                this->maxFoundPlainLength * this->activeHashesProcessed.size());
        printf("return code: %d\n", err);
        exit(1);
    }
    // Clear the host found password space.
    memset(this->HostFoundPasswordsAddress, 0,
            this->maxFoundPlainLength * this->activeHashesProcessed.size());

    if (this->useZeroCopy) {
        err = cudaHostGetDevicePointer((void **)&this->DeviceFoundPasswordsAddress,
            this->HostFoundPasswordsAddress, 0);
        err2 = cudaSuccess;
    } else {
        err = cudaMalloc((void **)&this->DeviceFoundPasswordsAddress,
            this->maxFoundPlainLength * this->activeHashesProcessed.size());
        err2 = cudaMemset(this->DeviceFoundPasswordsAddress, 0,
            this->maxFoundPlainLength * this->activeHashesProcessed.size());
    }
    if ((err != cudaSuccess) || (err2 != cudaSuccess)) {
        printf("Unable to allocate %d bytes for device password list!  Exiting!\n",
                this->maxFoundPlainLength * this->activeHashesProcessed.size());
        printf("return code: %d\n", err);
        printf("return code: %d\n", err2);
        exit(1);
    }

    /**
     * Allocate space for host and device start positions.  To improve performance,
     * this space is now aligned for improved coalescing performance.  All the
     * position 0 elements are together, followed by all the position 1 elements,
     * etc.
     *
     * This memory can be allocated as write combined, as it is not read by
     * the host ever - only written.  Since it is regularly transferred to the
     * GPU, this should help improve performance.
     */
    err = cudaHostAlloc((void**)&this->HostStartPointAddress,
        this->TotalKernelWidth * this->passwordLength,
        cudaHostAllocWriteCombined | cudaHostMallocFlags);

    err2 = cudaMalloc((void **)&this->DeviceStartPointAddress,
        this->TotalKernelWidth * this->passwordLength);

    if ((err != cudaSuccess) || (err2 != cudaSuccess)) {
        printf("Unable to allocate %d bytes for host/device startpos list!  Exiting!\n",
                this->TotalKernelWidth * this->passwordLength);
        printf("return code: %d\n", err);
        printf("return code: %d\n", err2);
        exit(1);
    }
    
    /**
     * Allocate space for the device start password values.  This is a copy of
     * the MFNHashTypePlain::HostStartPasswords32 vector for the GPU.
     */
    err = cudaMalloc((void **)&this->DeviceStartPasswords32Address,
        this->TotalKernelWidth * this->passwordLengthWords);
    
    if ((err != cudaSuccess)) {
        printf("Unable to allocate %d bytes for host/device startpos list!  Exiting!\n",
                this->TotalKernelWidth * this->passwordLengthWords);
        printf("return code: %d\n", err);
        exit(1);
    }

    err = cudaMalloc((void **)&this->DeviceBitmap256kb_Address,
        256 * 1024);
    if (err == cudaSuccess) {
        memalloc_printf("Successfully allocated Bitmap 256kb\n");
    } else {
        memalloc_printf("Unable to allocate Bitmap 256kb\n");
        this->DeviceBitmap256kb_Address = 0;
        cudaGetLastError();
    }
    
    // If a wordlist is requested, attempt to allocate 128MB of wordlist.
    if (this->hashAttributes.hashUsesWordlist) {
        err = cudaMalloc((void **)&this->DeviceWordlistBlocks,
            (128 * 1024 * 1024));
        if (err != cudaSuccess) {
            printf("Unable to allocate %d bytes for wordlist 128MB array!  Exiting!\n");
            printf("return code: %d\n", err);
            exit(1);
        }

        // And allocate 32M worth of space for lengths - should be enough for
        // 128MB of 4-byte words.
        err = cudaMalloc((void **)&this->DeviceWordlistLengths,
            (32 * 1024 * 1024));
        if (err != cudaSuccess) {
            printf("Unable to allocate %d bytes for wordlist 32MB array!  Exiting!\n");
            printf("return code: %d\n", err);
            exit(1);
        }
    }
    
    // If the hash is salted, allocate salt space.
    if (this->hashAttributes.hashUsesSalt) {
        /**
        * Extend the MFNHashTypePlainCUDA class to allocate the memory for the salt
        * length array and the salt value array.  These allocation are done first,
        * because the base class will use the remaining memory for bitmaps, which
        * means that these allocations could fail.
        */

        /*
        * Malloc the device salt length array size.  This is a vector of
        * uint32_t values.
        */
        err = cudaMalloc((void **)&this->DeviceSaltLengthsAddress,
            this->saltLengths.size() * sizeof(uint32_t));
        if (err != cudaSuccess) {
            printf("Unable to allocate %d bytes for salt length array!  Exiting!\n",
                    this->saltLengths.size() * sizeof(uint32_t));
            printf("return code: %d\n", err);
            exit(1);
        }

        /*
        * Malloc the device salt value array size.  This is a vector of
        * uint32_t values.
        */
        err = cudaMalloc((void **)&this->DeviceSaltValuesAddress,
            this->activeSaltsDeviceformat.size() * sizeof(uint32_t));
        if (err != cudaSuccess) {
            printf("Unable to allocate %d bytes for salt values array!  Exiting!\n",
                    this->activeSaltsDeviceformat.size() * sizeof(uint32_t));
            printf("return code: %d\n", err);
            exit(1);
        }        
    }
    
    /**
     * Finally, attempt to allocate space for the giant device bitmaps.  There
     * are 4x128MB bitmaps, and any number can be allocated.  If they are not
     * fully allocated, their address is set to null as a indicator to the device
     * that there is no data present.  Attempt to allocate as many as possible.
     *
     * This will be accessed regularly, so should probably not be zero copy.
     * Also, I'm not sure how mapping host memory into multiple threads would
     * work.  Typically, if the GPU doesn't have enough RAM for the full
     * set of bitmaps, it's a laptop, and therefore may be short on host RAM
     * for the pinned access.
     *
     * If there is an error in allocation, call cudaGetLastError() to clear it -
     * we know there has been an error, and do not want it to persist.
     */
    err = cudaMalloc((void **)&this->DeviceBitmap128mb_a_Address,
        128 * 1024 * 1024);
    if (err == cudaSuccess) {
        memalloc_printf("Successfully allocated Bitmap A\n");
    } else {
        memalloc_printf("Unable to allocate 128MB bitmap A\n");
        this->DeviceBitmap128mb_a_Address = 0;
        cudaGetLastError();
    }

    err = cudaMalloc((void **)&this->DeviceBitmap128mb_b_Address,
        128 * 1024 * 1024);
    if (err == cudaSuccess) {
        memalloc_printf("Successfully allocated Bitmap B\n");
    } else {
        memalloc_printf("Unable to allocate 128MB bitmap B\n");
        this->DeviceBitmap128mb_b_Address = 0;
        cudaGetLastError();
    }

    err = cudaMalloc((void **)&this->DeviceBitmap128mb_c_Address,
        128 * 1024 * 1024);
    if (err == cudaSuccess) {
        memalloc_printf("Successfully allocated Bitmap C\n");
    } else {
        memalloc_printf("Unable to allocate 128MB bitmap C\n");
        this->DeviceBitmap128mb_c_Address = 0;
        cudaGetLastError();
    }

    err = cudaMalloc((void **)&this->DeviceBitmap128mb_d_Address,
        128 * 1024 * 1024);
    if (err == cudaSuccess) {
        memalloc_printf("Successfully allocated Bitmap D\n");
    } else {
        memalloc_printf("Unable to allocate 128MB bitmap D\n");
        this->DeviceBitmap128mb_d_Address = 0;
        cudaGetLastError();
    }
    //printf("Thread %d memory allocated successfully\n", this->threadId);
}


void MFNHashTypePlainCUDA::freeThreadAndDeviceMemory() {
    trace_printf("MFNHashTypePlainCUDA::freeThreadAndDeviceMemory()\n");

    cudaError_t err;

    // Free all the memory, then look for errors.
    cudaFree((void *)this->DeviceHashlistAddress);
    cudaFreeHost((void *)this->HostSuccessAddress);

    delete[] this->HostSuccessReportedAddress;

    // Only cudaFree if zeroCopy is in use.
    if (!this->useZeroCopy) {
        cudaFree((void *)this->DeviceSuccessAddress);
        cudaFree((void *)this->DeviceFoundPasswordsAddress);

    }
    
    cudaFreeHost((void *)this->HostFoundPasswordsAddress);

    cudaFreeHost((void*)this->HostStartPointAddress);
    cudaFree((void *)this->DeviceStartPointAddress);
    cudaFree((void *)this->DeviceStartPasswords32Address);

    // Free salted hashes if in use.
    if (this->hashAttributes.hashUsesWordlist) {
        cudaFree((void *)this->DeviceWordlistBlocks);
        cudaFree((void *)this->DeviceWordlistLengths);
    }

    if (this->hashAttributes.hashUsesSalt) {
        cudaFree((void *)this->DeviceSaltLengthsAddress);
        cudaFree((void *)this->DeviceSaltValuesAddress);
    }
    
    // Only free the bitmap memory if it has been allocated.
    if (this->DeviceBitmap256kb_Address) {
        cudaFree((void *)this->DeviceBitmap256kb_Address);
        this->DeviceBitmap256kb_Address = 0;
    }
    if (this->DeviceBitmap128mb_a_Address) {
        cudaFree((void *)this->DeviceBitmap128mb_a_Address);
        this->DeviceBitmap128mb_a_Address = 0;
    }
    if (this->DeviceBitmap128mb_b_Address) {
        cudaFree((void *)this->DeviceBitmap128mb_b_Address);
        this->DeviceBitmap128mb_b_Address = 0;
    }
    if (this->DeviceBitmap128mb_c_Address) {
        cudaFree((void *)this->DeviceBitmap128mb_c_Address);
        this->DeviceBitmap128mb_c_Address = 0;
    }
    if (this->DeviceBitmap128mb_d_Address) {
        cudaFree((void *)this->DeviceBitmap128mb_d_Address);
        this->DeviceBitmap128mb_d_Address = 0;
    }

    // Get any error that occurred above and report it.
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d: CUDA error freeing memory: %s. Exiting.\n",
                this->threadId, cudaGetErrorString( err));
        exit(1);
    }
}


void MFNHashTypePlainCUDA::copyDataToDevice() {
    trace_printf("MFNHashTypePlainCUDA::copyDataToDevice()\n");
    cudaError_t err;

    // Copy all the various elements of data to the device, forming them as needed.
    if (this->hashAttributes.hashUsesSalt) {
        this->copySaltArraysToDevice();
    }
    
    // Device hashlist: Copy hashlist to device.
    err = cudaMemcpy(this->DeviceHashlistAddress,
            &this->activeHashesProcessedDeviceformat[0],
            this->activeHashesProcessedDeviceformat.size(),
            cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Thread %d: Cannot copy hashlist to GPU memory!.\n",
                this->threadId);
        exit(1);
    }

    // Device bitmaps: Copy all relevant bitmaps to the device.
    // Only copy bitmaps that are created.
    if (this->DeviceBitmap256kb_Address) {
        memalloc_printf("Thread %d: Copying bitmap 256kb\n", this->threadId);
        err = cudaMemcpy(this->DeviceBitmap256kb_Address,
                &this->globalBitmap256kb_a[0],
                this->globalBitmap256kb_a.size(),
                cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Thread %d: Cannot copy bitmap 256kb to GPU memory!.\n",
                    this->threadId);
            exit(1);
        }
    }

    if (this->DeviceBitmap128mb_a_Address) {
        memalloc_printf("Thread %d: Copying bitmap A\n", this->threadId);
        err = cudaMemcpy(this->DeviceBitmap128mb_a_Address,
                &this->globalBitmap128mb_a[0],
                this->globalBitmap128mb_a.size(),
                cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Thread %d: Cannot copy bitmap A to GPU memory!.\n",
                    this->threadId);
            exit(1);
        }
    }

    if (this->DeviceBitmap128mb_b_Address) {
        memalloc_printf("Thread %d: Copying bitmap B\n", this->threadId);
        err = cudaMemcpy(this->DeviceBitmap128mb_b_Address,
                &this->globalBitmap128mb_b[0],
                this->globalBitmap128mb_b.size(),
                cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Thread %d: Cannot copy bitmap B to GPU memory!.\n",
                    this->threadId);
            exit(1);
        }
    }

    if (this->DeviceBitmap128mb_c_Address) {
        memalloc_printf("Thread %d: Copying bitmap C\n", this->threadId);
        err = cudaMemcpy(this->DeviceBitmap128mb_c_Address,
                &this->globalBitmap128mb_c[0],
                this->globalBitmap128mb_c.size(),
                cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Thread %d: Cannot copy bitmap C to GPU memory!.\n",
                    this->threadId);
            exit(1);
        }
    }

    if (this->DeviceBitmap128mb_d_Address) {
        memalloc_printf("Thread %d: Copying bitmap D\n", this->threadId);
        err = cudaMemcpy(this->DeviceBitmap128mb_d_Address,
                &this->globalBitmap128mb_d[0],
                this->globalBitmap128mb_d.size(),
                cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Thread %d: Cannot copy bitmap D to GPU memory!.\n",
                    this->threadId);
            exit(1);
        }
    }
    // Other data to the device - charset, etc.
}

void MFNHashTypePlainCUDA::copyWordlistToDevice(
        std::vector <uint8_t> &wordlistLengths,
        std::vector<uint32_t> &wordlistData) {
    trace_printf("MFNHashTypePlainCUDA::copyWordlistToDevice()\n");

    cudaError_t err;
    uint32_t wordCount;
    uint8_t blocksPerWord;

    // Copy the bytes of wordlist length.
    err = cudaMemcpy(this->DeviceWordlistLengths,
            &wordlistLengths[0],
            wordlistLengths.size(),
            cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Thread %d: Cannot copy wordlist lengths to GPU memory!.\n",
                this->threadId);
        printf("Trying to copy %lu bytes\n", wordlistLengths.size());
        exit(1);
    }

    // And the words of data.
    err = cudaMemcpy(this->DeviceWordlistBlocks,
            &wordlistData[0],
            wordlistData.size() * 4,
            cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Thread %d: Cannot copy wordlist blocks to GPU memory!.\n",
                this->threadId);
        printf("Trying to copy %lu bytes\n", wordlistData.size() * 4);
        exit(1);
    }

    // Set the wordlist size
    wordCount = wordlistLengths.size();
    // Determine the blocks-per-word
    blocksPerWord = wordlistData.size() / wordlistLengths.size();

    this->copyWordlistSizeToDevice(wordCount, blocksPerWord);
}

void MFNHashTypePlainCUDA::copyStartPointsToDevice() {
    trace_printf("MFNHashTypePlainCUDA::copyStartPointsToDevice()\n");
    cudaMemcpy(this->DeviceStartPointAddress,
               this->HostStartPointAddress,
               this->TotalKernelWidth * this->passwordLength,
               cudaMemcpyHostToDevice);

    cudaMemcpy(this->DeviceStartPasswords32Address,
               &this->HostStartPasswords32[0],
               this->TotalKernelWidth * this->passwordLengthWords,
               cudaMemcpyHostToDevice);
}


int MFNHashTypePlainCUDA::setCUDADeviceID(int newCUDADeviceId) {
    trace_printf("MFNHashTypePlainCUDA::setCUDADeviceID(%d)\n", newCUDADeviceId);
    
    CHCUDAUtils *CudaUtils = MultiforcerGlobalClassFactory.getCudaUtilsClass();
    MFNCommandLineData *CommandLineData = MultiforcerGlobalClassFactory.getCommandlinedataClass();
    
    if (newCUDADeviceId >= CudaUtils->getCudaDeviceCount()) {
        printf("Invalid device ID - greater than number of devices!\n");
        exit(1);
    }
    
    this->gpuDeviceId = newCUDADeviceId;
    
    // If the blocks or threads are set, use them, else use the default.
    if (CommandLineData->GetGpuBlocks()) {
        this->GPUBlocks = CommandLineData->GetGpuBlocks();
        klaunch_printf("Using CLI GPU Blocks %d\n", this->GPUBlocks);
    } else {
        this->GPUBlocks = CudaUtils->getCudaDefaultBlockCount(newCUDADeviceId);
        klaunch_printf("Using Default GPU Blocks %d\n", this->GPUBlocks);
    }
    
    if (CommandLineData->GetGpuThreads()) {
        this->GPUThreads = CommandLineData->GetGpuThreads();
        klaunch_printf("Using CLI GPU Threads %d\n", this->GPUThreads);
    } else {
        this->GPUThreads = CudaUtils->getCudaDefaultThreadCount(newCUDADeviceId);
        klaunch_printf("Using Default GPU Threads %d\n", this->GPUThreads);
    }
    
    // If target time is 0, use defaults.
    if (CommandLineData->GetTargetExecutionTimeMs()) {
        this->kernelTimeMs = CommandLineData->GetTargetExecutionTimeMs();
    } else {
        if (CudaUtils->getCudaHasTimeout(newCUDADeviceId)) {
            this->kernelTimeMs = 50;
        } else {
            this->kernelTimeMs = 500;
        }
    }
    
    // Override thread count if needed for hash type.
    this->GPUThreads = this->getMaxHardwareThreads(this->GPUThreads);
    
    this->VectorWidth = 1;
    this->TotalKernelWidth = this->GPUBlocks * this->GPUThreads * this->VectorWidth;
    
    //printf("Successfully added device %d, thread ID %d\n", newCUDADeviceId, this->threadId);
    //printf("Thread %d blocks/threads/vec: %d/%d/%d\n", this->threadId, this->GPUBlocks, this->GPUThreads, this->VectorWidth);

    return 1;
}

void MFNHashTypePlainCUDA::setupClassForMultithreadedEntry() {
    trace_printf("MFNHashTypePlainCUDA::setupClassForMultithreadedEntry()\n");
}

void MFNHashTypePlainCUDA::synchronizeThreads() {
    cudaThreadSynchronize();
}


void MFNHashTypePlainCUDA::setStartPoints(uint64_t perThread, uint64_t startPoint) {
    trace_printf("MFNHashTypePlain::setStartPoints()\n");

    uint32_t numberThreads = this->TotalKernelWidth;
    uint64_t threadId, threadStartPoint;
    uint32_t characterPosition;

    uint8_t *threadStartPosition = this->HostStartPointAddress;

    if (this->isSingleCharset) {
        klaunch_printf("Calculating start points for a single charset.\n");
        // Copy the current charset length into a local variable for speed.
        uint8_t currentCharsetLength = this->currentCharset.at(0).size();

        for (threadId = 0; threadId < numberThreads; threadId++) {
            threadStartPoint = threadId * perThread + startPoint;
            //printf("Thread %u, startpoint %lu\n", threadId, threadStartPoint);

            // Loop through all the character positions.  This is easier than a case statement.
            for (characterPosition = 0; characterPosition < this->passwordLength; characterPosition++) {
                threadStartPosition[characterPosition * numberThreads + threadId] =
                        (uint8_t)(threadStartPoint % currentCharsetLength);
                threadStartPoint /= currentCharsetLength;
                /*printf("Set thread %d to startpoint %d at pos %d\n",
                        threadId, threadStartPosition[characterPosition * numberThreads + threadId],
                        characterPosition * numberThreads + threadId);*/
            }
        }

    } else{
        klaunch_printf("Calculating start points for a multiple charset.\n");
        if (this->passwordLength > this->currentCharset.size()) {
            printf("Error: Password length > charset length!\n");
            printf("Terminating!\n");
            exit(1);
        }
        for (threadId = 0; threadId < numberThreads; threadId++) {
            threadStartPoint = threadId * perThread + startPoint;
            //printf("Thread %u, startpoint %lu\n", threadId, threadStartPoint);

            // Loop through all the character positions.  This is easier than a case statement.
            for (characterPosition = 0; characterPosition < this->passwordLength; characterPosition++) {
                threadStartPosition[characterPosition * numberThreads + threadId] =
                        (uint8_t)(threadStartPoint % this->currentCharset[characterPosition].size());
                threadStartPoint /= this->currentCharset[characterPosition].size();
                /*printf("Set thread %d to startpoint %d at pos %d\n",
                        threadId, threadStartPosition[characterPosition * numberThreads + threadId],
                        characterPosition * numberThreads + threadId);*/
            }
        }
    }
}


void MFNHashTypePlainCUDA::copyDeviceFoundPasswordsToHost() {
    trace_printf("MFNHashTypePlainCUDA::copyDeviceFoundPasswordsToHost()\n");

    cudaError_t err, err2;

    err = cudaMemcpy(this->HostSuccessAddress,
            this->DeviceSuccessAddress,
            this->activeHashesProcessed.size(), cudaMemcpyDeviceToHost);
    
    err2 = cudaMemcpy(this->HostFoundPasswordsAddress,
            this->DeviceFoundPasswordsAddress,
            this->maxFoundPlainLength * this->activeHashesProcessed.size(), cudaMemcpyDeviceToHost);

    if ((err != cudaSuccess) || (err2 != cudaSuccess)) {
        printf("Error copying passwords from device to host!\n");
        printf("err: %d, err2: %d\n", err, err2);
        exit(1);
    }

}

void MFNHashTypePlainCUDA::outputFoundHashes() {
    trace_printf("MFNHashTypePlain::outputFoundHashes()\n");
    uint32_t i, j;

    /**
     * A vector containing the hash, processed back into the raw format.
     */
    std::vector<uint8_t> rawHash;
    std::vector<uint8_t> foundPassword;

    uint8_t *hostSuccessArray = this->HostSuccessAddress;
    uint8_t *hostSuccessReportedArray = this->HostSuccessReportedAddress;
    uint8_t *hostPasswords = this->HostFoundPasswordsAddress;

    for (i = 0; i < this->activeHashesProcessed.size(); i++) {
        if (hostSuccessArray[i] && !hostSuccessReportedArray[i]) {
            rawHash = this->postProcessHash(this->activeHashesProcessed[i]);
            // Resize to the max pass length + 1 - this ensures that strlen
            // will find a null byte at the end and stop measuring.
            foundPassword.resize(this->maxFoundPlainLength + 1, 0);
            for (j = 0; j < this->maxFoundPlainLength; j++) {
                foundPassword[j] = hostPasswords[this->maxFoundPlainLength * i + j];
            }
            // Resize to the length of the found password.
            foundPassword.resize(strlen((char *)&foundPassword[0]));
            this->HashFile->ReportFoundPassword(rawHash, foundPassword, hostSuccessArray[i]);
            // Report the found hash over the network.
            if (this->CommandLineData->GetIsNetworkClient()) {
                MultiforcerGlobalClassFactory.getNetworkClientClass()->
                        submitFoundHash(rawHash, foundPassword, hostSuccessArray[i]);
            }
            this->Display->addCrackedPassword(foundPassword);
            hostSuccessReportedArray[i] = 1;
        }
    }

    // Check to see if we should exit (as all hashes are found).
    if (this->HashFile->GetUncrackedHashCount() == 0) {
      //global_interface.exit = 1;
    }
}


void MFNHashTypePlainCUDA::copySaltArraysToDevice() {
    trace_printf("MFNHashTypeSaltedCUDA::copySaltArraysToDevice()\n");
    
    cudaError_t err;

    // Device salt lengths.
    err = cudaMemcpy(this->DeviceSaltLengthsAddress,
            &this->saltLengths[0],
            this->saltLengths.size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Thread %d: Cannot copy salt length list to GPU memory!.\n",
                this->threadId);
        exit(1);
    }

    // Device salt values.
    err = cudaMemcpy(this->DeviceSaltValuesAddress,
            &this->activeSaltsDeviceformat[0],
            this->activeSaltsDeviceformat.size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Thread %d: Cannot copy salt values to GPU memory!.\n",
                this->threadId);
        exit(1);
    }

    numberSaltsCopiedToDevice = this->activeSalts.size();
    // Set the more global value as well.
    this->numberUniqueSalts = numberSaltsCopiedToDevice;

    // Push the updated numbers into constant.
    this->copySaltConstantsToDevice();
}