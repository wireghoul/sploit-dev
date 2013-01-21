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

#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL.h"
#include "MFN_Common/MFNCommandLineData.h"
#include "MFN_Common/MFNMultiforcerClassFactory.h"
#include "CH_HashFiles/CHHashFileVPlain.h"
#include "MFN_Common/MFNDisplay.h"
#include "OpenCL_Common/GRTOpenCL.h"
#include "GRT_OpenCL_host/GRTCLUtils.h"
#include "MFN_Common/MFNDebugging.h"
#include "MFN_Common/MFNDefines.h"
#include "MFN_Common/MFNNetworkClient.h"


#define MFN_HASH_TYPE_PLAIN_CUDA_MD5_MAX_CHARSET_LENGTH 128

extern MFNClassFactory MultiforcerGlobalClassFactory;


MFNHashTypePlainOpenCL::MFNHashTypePlainOpenCL(int hashLengthBytes) :  MFNHashTypePlain(hashLengthBytes) {
    trace_printf("MFNHashTypePlainOpenCL::MFNHashTypePlainOpenCL(%d)\n", hashLengthBytes);

    this->MFNHashTypeMutex.lock();
    this->threadId = MultiforcerGlobalClassFactory.getDisplayClass()->getFreeThreadId(GPU_THREAD);
    this->numberThreads++;
    trace_printf("MFNHashType GPU/OpenCL Thread ID %d\n", this->threadId);
    this->MFNHashTypeMutex.unlock();
}

MFNHashTypePlainOpenCL::~MFNHashTypePlainOpenCL() {
    trace_printf("MFNHashTypePlainOpenCL::~MFNHashTypePlainOpenCL()\n");
    delete this->OpenCL;
}

void MFNHashTypePlainOpenCL::setupDevice() {
    trace_printf("CHHashTypeVPlainOpenCL::setupDevice()\n");

    // Set the OpenCL platform & device
    trace_printf("Thread %d setting OpenCL platform/device to %d, %d\n",
            this->threadId, this->openCLPlatformId, this->gpuDeviceId);
    this->OpenCL->selectPlatformById(this->openCLPlatformId);
    this->OpenCL->selectDeviceById(this->gpuDeviceId);
    
    // Store the total memory size in bytes.
    this->DeviceAvailableMemoryBytes = this->OpenCL->getGlobalMemorySize();
    
}

void MFNHashTypePlainOpenCL::doKernelSetup() {
    trace_printf("CHHashTypeVPlainOpenCL::doKernelSetup()\n");
    char buildOptions[1024];
    cl_int errorCode;

    /**
     * Handle generating the kernels.  This involves building with the specified
     * password length, vector width, and BFI_INT status.
     */
    

    if (MultiforcerGlobalClassFactory.getCommandlinedataClass()->GetUseBfiInt()) {
        // BFI_INT patching - pass BITALIGN to kernel
        sprintf(buildOptions, "-D PASSWORD_LENGTH=%d -D VECTOR_WIDTH=%d -D BITALIGN=1 -D THREADSPERBLOCK=%d -D SHARED_BITMAP_SIZE=%d",
            this->passwordLength, this->VectorWidth, this->GPUThreads, this->sharedBitmapSize);
    } else {
        // No BFI_INT patching.
        sprintf(buildOptions, "-D PASSWORD_LENGTH=%d -D VECTOR_WIDTH=%d -D THREADSPERBLOCK=%d -D SHARED_BITMAP_SIZE=%d",
                this->passwordLength, this->VectorWidth, this->GPUThreads, this->sharedBitmapSize);
    }
    // Compile the code using additional define strings as needed by the hash type.
    this->OpenCL->buildProgramFromManySourcesConcat(this->getHashFileNames(), buildOptions, 
            this->getDefineStrings() + this->getKernelSourceString());

    // If the BFI_INT patching is being used, patch the generated binary.
    if (MultiforcerGlobalClassFactory.getCommandlinedataClass()->GetUseBfiInt()) {
        this->OpenCL->doAMDBFIPatch();
    }

    this->HashProgram = this->OpenCL->getProgram();
    
    // Create a kernel for one-kernel hash types if present
    if (this->getHashKernelName().length()) {
        this->HashKernel = clCreateKernel (this->HashProgram,
                this->getHashKernelName().c_str(), &errorCode);
    }

    // Create the kernels for multi-kernel hash types
    {
        this->HashKernelVector.clear();
        std::vector<std::string> hashNames = this->getHashKernelNamesVector();
        for (int i = 0; i < hashNames.size(); i++) {
            klaunch_printf("Trying to build kernel %s\n", hashNames[i].c_str());
            this->HashKernelVector.push_back(
                    clCreateKernel (this->HashProgram, hashNames[i].c_str(), 
                    &errorCode));
            // Check for errors building the kernel.
            if (errorCode != CL_SUCCESS) {
                printf("Error: %s\n", print_cl_errstring(errorCode));
                exit(1);
            }
        }
    }
    
    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }
}

void MFNHashTypePlainOpenCL::teardownDevice() {
    trace_printf("MFNHashTypePlainOpenCL::teardownDevice()\n");
}

void MFNHashTypePlainOpenCL::allocateThreadAndDeviceMemory() {
    trace_printf("MFNHashTypePlainOpenCL::allocateThreadAndDeviceMemory()\n");

    /**
     * Error variable - stores the result of the various mallocs & such.
     */
    cl_int errorCode;
    /*
     * Malloc the device hashlist space.  This is the number of available hashes
     * times the hash length in bytes.  The data will be copied later.
     */
    memalloc_printf("Attempting to openclMalloc %d bytes for device hashlist for thread %d.\n",
            this->activeHashesProcessed.size() * this->hashLengthBytes, this->threadId);
    this->DeviceHashlistAddress =
            clCreateBuffer (this->OpenCL->getContext(),
            CL_MEM_READ_ONLY,
            this->activeHashesProcessed.size() * this->hashLengthBytes,
            NULL,
            &errorCode);
    this->DeviceAvailableMemoryBytes -= this->activeHashesProcessed.size() *
            this->hashLengthBytes;
    if (errorCode != CL_SUCCESS) {
        printf("Unable to allocate %d bytes for device hashlist!  Exiting!\n",
                this->activeHashesProcessed.size() * this->hashLengthBytes);
        printf("Error: %s\n", print_cl_errstring(errorCode));
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
    memalloc_printf("Attempting to cudaHostAlloc %d bytes for HostSuccess\n",
            this->activeHashesProcessed.size());
    this->HostSuccessAddress = new uint8_t [this->activeHashesProcessed.size()];
    memset(this->HostSuccessAddress, 0, this->activeHashesProcessed.size());

    // Allocate memory for the reported flags.
    this->HostSuccessReportedAddress = new uint8_t [this->activeHashesProcessed.size()];
    memset(this->HostSuccessReportedAddress, 0, this->activeHashesProcessed.size());

    // Allocate device memory for the "reported" flags, and copy in the zeroed 
    // host memory for this region.
    this->DeviceSuccessAddress =
            clCreateBuffer (this->OpenCL->getContext(),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            this->activeHashesProcessed.size(),
            this->HostSuccessAddress,
            &errorCode);
    this->DeviceAvailableMemoryBytes -= this->activeHashesProcessed.size();
    if (errorCode != CL_SUCCESS) {
        printf("Unable to allocate %d bytes for device successlist!  Exiting!\n",
                this->activeHashesProcessed.size());
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    /*
     * Allocate memory for the found passwords.  As this is commonly copied
     * back and forth, it should be made zero copy if requested.
     *
     * This requires (number hashes * passwordLength) bytes of data.
     */

    this->HostFoundPasswordsAddress = new uint8_t [this->maxFoundPlainLength * 
            this->activeHashesProcessed.size()];
    // Clear the host found password space.
    memset(this->HostFoundPasswordsAddress, 0,
            this->maxFoundPlainLength * this->activeHashesProcessed.size());

    this->DeviceFoundPasswordsAddress =
            clCreateBuffer (this->OpenCL->getContext(),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            this->maxFoundPlainLength * this->activeHashesProcessed.size(),
            this->HostFoundPasswordsAddress,
            &errorCode);
    this->DeviceAvailableMemoryBytes -= this->maxFoundPlainLength *
            this->activeHashesProcessed.size();
    if (errorCode != CL_SUCCESS) {
        printf("Unable to allocate %d bytes for device passwordlist!  Exiting!\n",
                this->maxFoundPlainLength * this->activeHashesProcessed.size());
        printf("Error: %s\n", print_cl_errstring(errorCode));
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

    this->HostStartPointAddress = new uint8_t [this->TotalKernelWidth * 
            this->passwordLength];

    this->DeviceStartPointAddress =
            clCreateBuffer (this->OpenCL->getContext(),
            CL_MEM_READ_ONLY,
            this->TotalKernelWidth * this->passwordLength,
            NULL,
            &errorCode);
    this->DeviceAvailableMemoryBytes -= this->TotalKernelWidth * this->passwordLength;
    if (errorCode != CL_SUCCESS) {
        printf("Unable to allocate %d bytes for device start points!  Exiting!\n",
                this->TotalKernelWidth * this->passwordLength);
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }
    
    this->DeviceStartPasswords32Address =
            clCreateBuffer (this->OpenCL->getContext(),
            CL_MEM_READ_ONLY,
            this->TotalKernelWidth * this->passwordLengthWords,
            NULL,
            &errorCode);
    this->DeviceAvailableMemoryBytes -= this->TotalKernelWidth * this->passwordLengthWords;
    if (errorCode != CL_SUCCESS) {
        printf("Unable to allocate %d bytes for device start passwords!  Exiting!\n",
                this->TotalKernelWidth * this->passwordLengthWords);
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    /**
     * Allocate memory for the things that are considered constant in CUDA
     * and not stored in global memory.  For OpenCL, these are stored in a 
     * constant-tagged chunk of global memory (or something) and therefore
     * need to have space allocated in global memory.
     */
    
    this->DeviceBitmap8kb_Address =
            clCreateBuffer (this->OpenCL->getContext(),
            CL_MEM_READ_ONLY,
            8192,
            NULL,
            &errorCode);
    this->DeviceAvailableMemoryBytes -= 8192;
    if (errorCode == CL_SUCCESS) {
        memalloc_printf("Successfully allocated 8kb Bitmap A\n");
    } else {
        printf("Unable to allocate 8kb bitmap A\n");
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }
    
    this->DeviceBitmap16kb_Address =
            clCreateBuffer (this->OpenCL->getContext(),
            CL_MEM_READ_ONLY,
            16384,
            NULL,
            &errorCode);
    this->DeviceAvailableMemoryBytes -= 16384;
    if (errorCode == CL_SUCCESS) {
        memalloc_printf("Successfully allocated 16kb Bitmap A\n");
    } else {
        printf("Unable to allocate 16kb bitmap A\n");
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    this->DeviceBitmap32kb_Address =
            clCreateBuffer (this->OpenCL->getContext(),
            CL_MEM_READ_ONLY,
            32768,
            NULL,
            &errorCode);
    this->DeviceAvailableMemoryBytes -= 32768;
    if (errorCode == CL_SUCCESS) {
        memalloc_printf("Successfully allocated 8kb Bitmap A\n");
    } else {
        printf("Unable to allocate 8kb bitmap A\n");
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }
    
    this->DeviceBitmap256kb_a_Address =
            clCreateBuffer (this->OpenCL->getContext(),
            CL_MEM_READ_ONLY,
            (256 * 1024),
            NULL,
            &errorCode);
    this->DeviceAvailableMemoryBytes -= (256*1024);
    if (errorCode == CL_SUCCESS) {
        memalloc_printf("Successfully allocated 256kb Bitmap A\n");
    } else {
        printf("Unable to allocate 256kb bitmap A\n");
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    this->DeviceForwardCharsetAddress =
            clCreateBuffer (this->OpenCL->getContext(),
            CL_MEM_READ_ONLY,
            MFN_HASH_TYPE_PLAIN_CUDA_MD5_MAX_CHARSET_LENGTH * this->passwordLength,
            NULL,
            &errorCode);
    this->DeviceAvailableMemoryBytes -= 
            MFN_HASH_TYPE_PLAIN_CUDA_MD5_MAX_CHARSET_LENGTH * this->passwordLength;
    if (errorCode != CL_SUCCESS) {
        printf("Unable to allocate forward charset\n");
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }
    
    this->DeviceReverseCharsetAddress =
            clCreateBuffer (this->OpenCL->getContext(),
            CL_MEM_READ_ONLY,
            MFN_HASH_TYPE_PLAIN_CUDA_MD5_MAX_CHARSET_LENGTH * this->passwordLength,
            NULL,
            &errorCode);
    this->DeviceAvailableMemoryBytes -= 
            MFN_HASH_TYPE_PLAIN_CUDA_MD5_MAX_CHARSET_LENGTH * this->passwordLength;
    if (errorCode != CL_SUCCESS) {
        printf("Unable to allocate reverse charset\n");
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }
    
    this->DeviceCharsetLengthsAddress =
            clCreateBuffer (this->OpenCL->getContext(),
            CL_MEM_READ_ONLY,
            this->passwordLength,
            NULL,
            &errorCode);
    this->DeviceAvailableMemoryBytes -= this->passwordLength;
    if (errorCode != CL_SUCCESS) {
        printf("Unable to allocate charset lengths\n");
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }
    
    // If a wordlist is requested, attempt to allocate 128MB of wordlist.
    if (this->hashAttributes.hashUsesWordlist) {
        this->DeviceWordlistBlocks =
                clCreateBuffer (this->OpenCL->getContext(),
                CL_MEM_READ_ONLY,
                (128 * 1024 * 1024),
                NULL,
                &errorCode);
        this->DeviceAvailableMemoryBytes -= (128 * 1024 * 1024);
        if (errorCode != CL_SUCCESS) {
            printf("Unable to allocate wordlist\n");
            printf("Error: %s\n", print_cl_errstring(errorCode));
            exit(1);
        }
        // And allocate 32M worth of space for lengths - should be enough for
        // 128MB of 4-byte words.
        this->DeviceWordlistLengths =
                clCreateBuffer (this->OpenCL->getContext(),
                CL_MEM_READ_ONLY,
                (32 * 1024 * 1024),
                NULL,
                &errorCode);
        this->DeviceAvailableMemoryBytes -= (4 * 1024 * 1024);
        if (errorCode != CL_SUCCESS) {
            printf("Unable to allocate wordlist lengths\n");
            printf("Error: %s\n", print_cl_errstring(errorCode));
            exit(1);
        }
    }

    // Salted hash support.
    if (this->hashAttributes.hashUsesSalt) {
        /**
        * Extend the MFNHashTypePlainOpenCL class to allocate the memory for the salt
        * length array and the salt value array.  These allocation are done first,
        * because the base class will use the remaining memory for bitmaps, which
        * means that these allocations could fail.
        */

        /*
        * Malloc the device salt length array size.  This is a vector of
        * uint32_t values.
        */
        memalloc_printf("Attempting to openclMalloc %d bytes for device salt length array for thread %d.\n",
                this->saltLengths.size() * sizeof(uint32_t), this->threadId);
        this->DeviceSaltLengthsAddress =
                clCreateBuffer (this->OpenCL->getContext(),
                CL_MEM_READ_ONLY,
                this->saltLengths.size() * sizeof(uint32_t),
                NULL,
                &errorCode);
        this->DeviceAvailableMemoryBytes -= this->saltLengths.size() * sizeof(uint32_t);
        if (errorCode != CL_SUCCESS) {
            printf("Unable to allocate %d bytes for device salts!  Exiting!\n",
                    this->saltLengths.size() * sizeof(uint32_t));
            printf("Error: %s\n", print_cl_errstring(errorCode));
            exit(1);
        }

        /*
        * Malloc the device salt value array size.  This is a vector of
        * uint32_t values.
        */
        memalloc_printf("Attempting to openclMalloc %d bytes for device salt value array for thread %d.\n",
                this->activeSaltsDeviceformat.size() * sizeof(uint32_t), this->threadId);
        this->DeviceSaltValuesAddress =
                clCreateBuffer (this->OpenCL->getContext(),
                CL_MEM_READ_ONLY,
                this->activeSaltsDeviceformat.size() * sizeof(uint32_t),
                NULL,
                &errorCode);
        this->DeviceAvailableMemoryBytes -= this->activeSaltsDeviceformat.size() * sizeof(uint32_t);
        if (errorCode != CL_SUCCESS) {
            printf("Unable to allocate %d bytes for device hashlist!  Exiting!\n",
                    this->activeHashesProcessed.size() * this->hashLengthBytes);
            printf("Error: %s\n", print_cl_errstring(errorCode));
            exit(1);
        }
    }
    
    // Iteration count support
    if (this->hashAttributes.hashUsesIterationCount) {
        memalloc_printf("Attempting to openclMalloc %d bytes for device iteration counts for thread %d.\n",
                this->activeIterationCounts.size() * sizeof(uint32_t), this->threadId);
        this->DeviceIterationCountAddresses =
                clCreateBuffer (this->OpenCL->getContext(),
                CL_MEM_READ_ONLY,
                this->activeIterationCounts.size() * sizeof(uint32_t),
                NULL,
                &errorCode);
        this->DeviceAvailableMemoryBytes -= this->activeIterationCounts.size() * sizeof(uint32_t);
        if (errorCode != CL_SUCCESS) {
            printf("Unable to allocate %d bytes for device salts!  Exiting!\n",
                    this->activeIterationCounts.size() * sizeof(uint32_t));
            printf("Error: %s\n", print_cl_errstring(errorCode));
            exit(1);
        }
    }
    
    // Temp space support
    if (this->hashAttributes.hashTempSpaceBytesPerElement) {
        this->DeviceTempSpaceAddress =
                clCreateBuffer (this->OpenCL->getContext(),
                CL_MEM_READ_WRITE,
                this->TotalKernelWidth *
                    this->hashAttributes.hashTempSpaceBytesPerElement,
                NULL,
                &errorCode);
        this->DeviceAvailableMemoryBytes -= this->TotalKernelWidth *
                    this->hashAttributes.hashTempSpaceBytesPerElement;
        if (errorCode != CL_SUCCESS) {
            printf("Unable to allocate %d bytes for device tempspace!  Exiting!\n",
                    this->TotalKernelWidth *
                    this->hashAttributes.hashTempSpaceBytesPerElement);
            printf("Error: %s\n", print_cl_errstring(errorCode));
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
     */
    if (this->DeviceAvailableMemoryBytes >= 128*1024*1024) {
        this->DeviceBitmap128mb_a_Address =
                clCreateBuffer (this->OpenCL->getContext(),
                CL_MEM_READ_ONLY,
                128 * 1024 * 1024,
                NULL,
                &errorCode);
        if (errorCode == CL_SUCCESS) {
            memalloc_printf("Successfully allocated Bitmap A\n");
            memalloc_printf("Bitmap A Address: %16llx\n", this->DeviceBitmap128mb_a_Address);
            this->DeviceAvailableMemoryBytes -= 128 * 1024 * 1024;
        } else {
            memalloc_printf("Unable to allocate 128MB bitmap A\n");
            this->DeviceBitmap128mb_a_Address = 0;
        }
    } else {
        memalloc_printf("Not enough free GPU mem to attempt 128MB bitmap A alloc.\n");
        this->DeviceBitmap128mb_a_Address = 0;
    }
    
    if (this->DeviceAvailableMemoryBytes >= 128*1024*1024) {
        this->DeviceBitmap128mb_b_Address =
                clCreateBuffer (this->OpenCL->getContext(),
                CL_MEM_READ_ONLY,
                128 * 1024 * 1024,
                NULL,
                &errorCode);
        if (errorCode == CL_SUCCESS) {
            memalloc_printf("Successfully allocated Bitmap B\n");
            memalloc_printf("Bitmap B Address: %16llx\n", this->DeviceBitmap128mb_b_Address);
            this->DeviceAvailableMemoryBytes -= 128 * 1024 * 1024;
        } else {
            memalloc_printf("Unable to allocate 128MB bitmap B\n");
            this->DeviceBitmap128mb_b_Address = 0;
        }
    } else {
        memalloc_printf("Not enough free GPU mem to attempt 128MB bitmap B alloc.\n");
        this->DeviceBitmap128mb_b_Address = 0;
    }
    
    if (this->DeviceAvailableMemoryBytes >= 128*1024*1024) {
        this->DeviceBitmap128mb_c_Address =
                clCreateBuffer (this->OpenCL->getContext(),
                CL_MEM_READ_ONLY,
                128 * 1024 * 1024,
                NULL,
                &errorCode);
        if (errorCode == CL_SUCCESS) {
            memalloc_printf("Successfully allocated Bitmap C\n");
            this->DeviceAvailableMemoryBytes -= 128 * 1024 * 1024;
        } else {
            memalloc_printf("Unable to allocate 128MB bitmap C\n");
            this->DeviceBitmap128mb_c_Address = 0;
        }
    } else {
        memalloc_printf("Not enough free GPU mem to attempt 128MB bitmap C alloc.\n");
        this->DeviceBitmap128mb_c_Address = 0;
    }
    
    if (this->DeviceAvailableMemoryBytes >= 128*1024*1024) {
        this->DeviceBitmap128mb_d_Address =
                clCreateBuffer (this->OpenCL->getContext(),
                CL_MEM_READ_ONLY,
                128 * 1024 * 1024,
                NULL,
                &errorCode);
        if (errorCode == CL_SUCCESS) {
            memalloc_printf("Successfully allocated Bitmap D\n");
            this->DeviceAvailableMemoryBytes -= 128 * 1024 * 1024;
        } else {
            memalloc_printf("Unable to allocate 128MB bitmap D\n");
            this->DeviceBitmap128mb_d_Address = 0;
        }
    } else {
        memalloc_printf("Not enough free GPU mem to attempt 128MB bitmap D alloc.\n");
        this->DeviceBitmap128mb_d_Address = 0;
    }
    
    memalloc_printf("Thread %d memory allocated successfully\n", this->threadId);
}


void MFNHashTypePlainOpenCL::freeThreadAndDeviceMemory() {
    trace_printf("MFNHashTypePlainOpenCL::freeThreadAndDeviceMemory()\n");

    
    clReleaseMemObject(this->DeviceHashlistAddress);
    delete[] this->HostSuccessAddress;
    delete[] this->HostSuccessReportedAddress;
    clReleaseMemObject(this->DeviceSuccessAddress);
    delete[] this->HostFoundPasswordsAddress;
    clReleaseMemObject(this->DeviceFoundPasswordsAddress);
    delete[] this->HostStartPointAddress;
    clReleaseMemObject(this->DeviceStartPointAddress);
    clReleaseMemObject(this->DeviceStartPasswords32Address);
    
    clReleaseMemObject(this->DeviceBitmap8kb_Address);
    clReleaseMemObject(this->DeviceBitmap16kb_Address);
    clReleaseMemObject(this->DeviceBitmap32kb_Address);
    clReleaseMemObject(this->DeviceBitmap256kb_a_Address);
    clReleaseMemObject(this->DeviceForwardCharsetAddress);
    clReleaseMemObject(this->DeviceReverseCharsetAddress);
    clReleaseMemObject(this->DeviceCharsetLengthsAddress);

    if (this->hashAttributes.hashUsesWordlist) {
        clReleaseMemObject(this->DeviceWordlistBlocks);
        clReleaseMemObject(this->DeviceWordlistLengths);
    }
    
    if (this->hashAttributes.hashUsesSalt) {
        clReleaseMemObject(this->DeviceSaltLengthsAddress);
        clReleaseMemObject(this->DeviceSaltValuesAddress);
    }

    if (this->hashAttributes.hashUsesIterationCount) {
        clReleaseMemObject(this->DeviceIterationCountAddresses);
    }
    
    if (this->hashAttributes.hashTempSpaceBytesPerElement) {
        clReleaseMemObject(this->DeviceTempSpaceAddress);
    }
    
    // Only free the bitmap memory if it has been allocated.
    if (this->DeviceBitmap128mb_a_Address) {
        clReleaseMemObject(this->DeviceBitmap128mb_a_Address);
        this->DeviceBitmap128mb_a_Address = 0;
    }
    if (this->DeviceBitmap128mb_b_Address) {
        clReleaseMemObject(this->DeviceBitmap128mb_b_Address);
        this->DeviceBitmap128mb_b_Address = 0;
    }
    if (this->DeviceBitmap128mb_c_Address) {
        clReleaseMemObject(this->DeviceBitmap128mb_c_Address);
        this->DeviceBitmap128mb_c_Address = 0;
    }
    if (this->DeviceBitmap128mb_d_Address) {
        clReleaseMemObject(this->DeviceBitmap128mb_d_Address);
        this->DeviceBitmap128mb_d_Address = 0;
    }
}


void MFNHashTypePlainOpenCL::copyDataToDevice() {
    trace_printf("MFNHashTypePlainOpenCL::copyDataToDevice()\n");
    
    cl_int errorCode;
    

    
    // Copy all the various elements of data to the device, forming them as needed.
    if (this->hashAttributes.hashUsesSalt) {
        this->copySaltArraysToDevice();
    }

    errorCode = clEnqueueWriteBuffer (this->OpenCL->getCommandQueue(),
            this->DeviceHashlistAddress,
            CL_TRUE /* blocking write */,
            0 /* offset */,
            this->activeHashesProcessedDeviceformat.size() /* bytes to copy */,
            (void *)&this->activeHashesProcessedDeviceformat[0],
            NULL, NULL, NULL /* event list stuff */);
    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }
    
    // Device bitmaps: Copy all relevant bitmaps to the device.
    // Only copy bitmaps that are created.
    if (this->DeviceBitmap128mb_a_Address) {
        memalloc_printf("Thread %d: Copying bitmap A\n", this->threadId);
        errorCode = clEnqueueWriteBuffer (this->OpenCL->getCommandQueue(),
                this->DeviceBitmap128mb_a_Address,
                CL_TRUE /* blocking write */,
                0 /* offset */,
                this->globalBitmap128mb_a.size() /* bytes to copy */,
                (void *)&this->globalBitmap128mb_a[0],
                NULL, NULL, NULL /* event list stuff */);
        if (errorCode != CL_SUCCESS) {
            printf("Error: %s\n", print_cl_errstring(errorCode));
            exit(1);
        }
    }

    if (this->DeviceBitmap128mb_b_Address) {
        memalloc_printf("Thread %d: Copying bitmap B\n", this->threadId);
        errorCode = clEnqueueWriteBuffer (this->OpenCL->getCommandQueue(),
                this->DeviceBitmap128mb_b_Address,
                CL_TRUE /* blocking write */,
                0 /* offset */,
                this->globalBitmap128mb_b.size() /* bytes to copy */,
                (void *)&this->globalBitmap128mb_b[0],
                NULL, NULL, NULL /* event list stuff */);
        if (errorCode != CL_SUCCESS) {
            printf("Error: %s\n", print_cl_errstring(errorCode));
            exit(1);
        }
    }
    
    if (this->DeviceBitmap128mb_c_Address) {
        memalloc_printf("Thread %d: Copying bitmap C\n", this->threadId);
        errorCode = clEnqueueWriteBuffer (this->OpenCL->getCommandQueue(),
                this->DeviceBitmap128mb_c_Address,
                CL_TRUE /* blocking write */,
                0 /* offset */,
                this->globalBitmap128mb_c.size() /* bytes to copy */,
                (void *)&this->globalBitmap128mb_c[0],
                NULL, NULL, NULL /* event list stuff */);
        if (errorCode != CL_SUCCESS) {
            printf("Error: %s\n", print_cl_errstring(errorCode));
            exit(1);
        }
    }
    
    if (this->DeviceBitmap128mb_d_Address) {
        memalloc_printf("Thread %d: Copying bitmap D\n", this->threadId);
        errorCode = clEnqueueWriteBuffer (this->OpenCL->getCommandQueue(),
                this->DeviceBitmap128mb_d_Address,
                CL_TRUE /* blocking write */,
                0 /* offset */,
                this->globalBitmap128mb_d.size() /* bytes to copy */,
                (void *)&this->globalBitmap128mb_d[0],
                NULL, NULL, NULL /* event list stuff */);
        if (errorCode != CL_SUCCESS) {
            printf("Error: %s\n", print_cl_errstring(errorCode));
            exit(1);
        }
    }

    // Other data to the device - charset, etc.
}

void MFNHashTypePlainOpenCL::copyWordlistToDevice(
        std::vector <uint8_t> &wordlistLengths,
        std::vector<uint32_t> &wordlistData) {
    trace_printf("MFNHashTypePlainOpenCL::copyWordlistToDevice()\n");
    
    cl_int errorCode;
    cl_uint wordCount;
    cl_uchar blocksPerWord;

    // Copy the bytes of wordlist length.
    errorCode = clEnqueueWriteBuffer (this->OpenCL->getCommandQueue(),
            this->DeviceWordlistLengths,
            CL_TRUE /* blocking write */,
            0 /* offset */,
            wordlistLengths.size() /* bytes to copy */,
            (void *)&wordlistLengths[0],
            NULL, NULL, NULL /* event list stuff */);
    if (errorCode != CL_SUCCESS) {
        printf("Error copying wordlist lengths: %s\n", print_cl_errstring(errorCode));
        printf("Trying to copy %lu bytes\n", wordlistLengths.size());
        exit(1);
    }

    // And the words of data.
    errorCode = clEnqueueWriteBuffer (this->OpenCL->getCommandQueue(),
            this->DeviceWordlistBlocks,
            CL_TRUE /* blocking write */,
            0 /* offset */,
            wordlistData.size() * 4 /* bytes to copy */,
            (void *)&wordlistData[0],
            NULL, NULL, NULL /* event list stuff */);
    if (errorCode != CL_SUCCESS) {
        printf("Error copying wordlist blocks: %s\n", print_cl_errstring(errorCode));
        printf("Trying to copy %lu bytes\n", wordlistData.size() * 4);
        exit(1);
    }

    // Set the wordlist size
    wordCount = wordlistLengths.size();
    // Determine the blocks-per-word
    blocksPerWord = wordlistData.size() / wordlistLengths.size();
    /*
    errorCode = clSetKernelArg (this->getKernelToRun(), 19, sizeof(cl_uint), &wordCount);
    errorCode |= clSetKernelArg (this->getKernelToRun(), 21, sizeof(cl_uchar), &blocksPerWord);
    
    if (errorCode != CL_SUCCESS) {
        printf("Error 1: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }*/
    this->copyWordlistSizeToDevice(wordCount, blocksPerWord);
}

void MFNHashTypePlainOpenCL::copyStartPointsToDevice() {
    trace_printf("MFNHashTypePlainOpenCL::copyStartPointsToDevice()\n");
    
    cl_int errorCode;
    
    errorCode = clEnqueueWriteBuffer (this->OpenCL->getCommandQueue(),
                this->DeviceStartPointAddress,
                CL_TRUE /* blocking write */,
                0 /* offset */,
                this->TotalKernelWidth * this->passwordLength /* bytes to copy */,
                (void *)this->HostStartPointAddress,
                NULL, NULL, NULL /* event list stuff */);

    errorCode = clEnqueueWriteBuffer (this->OpenCL->getCommandQueue(),
                this->DeviceStartPasswords32Address,
                CL_TRUE /* blocking write */,
                0 /* offset */,
                this->TotalKernelWidth * this->passwordLengthWords /* bytes to copy */,
                (void *)&this->HostStartPasswords32[0],
                NULL, NULL, NULL /* event list stuff */);
}


int MFNHashTypePlainOpenCL::setOpenCLDeviceID(int newOpenCLPlatformId, int newOpenCLDeviceId) {
    trace_printf("MFNHashTypePlainOpenCL::setOpenCLDeviceID(%d, %d)\n", newOpenCLPlatformId, newOpenCLDeviceId);
    
    MFNCommandLineData *CommandLineData = MultiforcerGlobalClassFactory.getCommandlinedataClass();
    
    this->OpenCL = new CryptohazeOpenCL();
    
    if (CommandLineData->GetDebug()) {
        this->OpenCL->enableDumpSourceFile();
    }

    if (newOpenCLPlatformId > this->OpenCL->getNumberOfPlatforms()) {
        printf("Error: OpenCL Platform ID %d not valid!\n", newOpenCLPlatformId);
        exit(1);
    }

    this->OpenCL->selectPlatformById(newOpenCLPlatformId);

    if (newOpenCLDeviceId > this->OpenCL->getNumberOfDevices()) {
        printf("Error: OpenCL Device ID %d not valid!\n", newOpenCLDeviceId);
        exit(1);
    }

    this->OpenCL->selectDeviceById(newOpenCLDeviceId);

    this->openCLPlatformId = newOpenCLPlatformId;
    this->gpuDeviceId = newOpenCLDeviceId;

    // Set up the shared bitmap size here, since we need it for calculating
    // hardware thread count.
    if (this->CommandLineData->GetSharedBitmapSize() == 0) {
        this->sharedBitmapSize = 8;
    } else {
        this->sharedBitmapSize = this->CommandLineData->GetSharedBitmapSize();
        if ((this->sharedBitmapSize != 8) && (this->sharedBitmapSize != 16) && 
                (this->sharedBitmapSize != 32)) {
            printf("Invalid shared bitmap size!  Must be 8, 16, 32!\n");
            exit(1);
        }
    }
 
    // If the blocks or threads are set, use them, else use the default.
    if (CommandLineData->GetGpuBlocks()) {
        this->GPUBlocks = CommandLineData->GetGpuBlocks();
    } else {
        this->GPUBlocks = this->OpenCL->getDefaultBlockCount();
    }

    if (CommandLineData->GetGpuThreads()) {
        this->GPUThreads = CommandLineData->GetGpuThreads();
    } else {
        this->GPUThreads = this->OpenCL->getDefaultThreadCount();
    }

    // If target time is 0, use defaults.
    if (CommandLineData->GetTargetExecutionTimeMs()) {
        this->kernelTimeMs = CommandLineData->GetTargetExecutionTimeMs();
    } else {
        this->kernelTimeMs = 200;
    }

    this->OpenCL->createContext();
    this->OpenCL->createCommandQueue();
 
    // If the vector width is specified, use it - else use default 8.
    if (this->CommandLineData->GetVectorWidth()) {
        this->VectorWidth = this->CommandLineData->GetVectorWidth();
    } else {
        this->VectorWidth = 8;
    }

    // Override thread count if needed for hash type.
    this->GPUThreads = this->getMaxHardwareThreads(this->GPUThreads);

    this->TotalKernelWidth = this->GPUBlocks * this->GPUThreads * this->VectorWidth;

    trace_printf("Thread %d added OpenCL Device (%d, %d)\n", this->threadId,
            newOpenCLPlatformId, newOpenCLDeviceId);;

    return 1;
}

void MFNHashTypePlainOpenCL::setupClassForMultithreadedEntry() {
    trace_printf("MFNHashTypePlainOpenCL::setupClassForMultithreadedEntry()\n");
}

void MFNHashTypePlainOpenCL::synchronizeThreads() {
    trace_printf("MFNHashTypePlainOpenCL::synchronizeThreads()\n");
    clEnqueueBarrier(this->OpenCL->getCommandQueue());
}


void MFNHashTypePlainOpenCL::setStartPoints(uint64_t perThread, uint64_t startPoint) {
    trace_printf("MFNHashTypePlain::setStartPoints()\n");

    uint32_t numberThreads = this->TotalKernelWidth;
    uint64_t threadId, threadStartPoint;
    uint32_t characterPosition;

    uint8_t *threadStartCharacters = this->HostStartPointAddress;

    if (this->isSingleCharset) {
        klaunch_printf("Calculating start points for a single charset.\n");
        // Copy the current charset length into a local variable for speed.
        uint8_t currentCharsetLength = this->currentCharset.at(0).size();

        for (threadId = 0; threadId < numberThreads; threadId++) {
            threadStartPoint = threadId * perThread + startPoint;
            //printf("Thread %u, startpoint %lu, perThread %d\n", threadId, threadStartPoint, perThread);

            // Loop through all the character positions.  This is easier than a case statement.
            for (characterPosition = 0; characterPosition < this->passwordLength; characterPosition++) {
                threadStartCharacters[characterPosition * numberThreads + threadId] =
                        this->currentCharset[0][(uint8_t)(threadStartPoint % currentCharsetLength)];
                threadStartPoint /= currentCharsetLength;
                /*printf("Set thread %d to startpoint %c at pos %d\n",
                        threadId, threadStartCharacters[characterPosition * numberThreads + threadId],
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
                threadStartCharacters[characterPosition * numberThreads + threadId] =
                        this->currentCharset[characterPosition][(uint8_t)(threadStartPoint % this->currentCharset[characterPosition].size())];
                threadStartPoint /= this->currentCharset[characterPosition].size();
                /*printf("Set thread %d to startpoint %d at pos %d\n",
                        threadId, threadStartPosition[characterPosition * numberThreads + threadId],
                        characterPosition * numberThreads + threadId);*/
            }
        }
    }
}


void MFNHashTypePlainOpenCL::copyDeviceFoundPasswordsToHost() {
    trace_printf("MFNHashTypePlainOpenCL::copyDeviceFoundPasswordsToHost()\n");

    cl_int errorCode;
    
    errorCode = clEnqueueReadBuffer (this->OpenCL->getCommandQueue(),
        this->DeviceSuccessAddress,
        CL_TRUE /* blocking write */,
        0 /* offset */,
        this->activeHashesProcessed.size() /* bytes to copy */,
        (void *)this->HostSuccessAddress,
        NULL, NULL, NULL /* event list stuff */);
    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    errorCode = clEnqueueReadBuffer (this->OpenCL->getCommandQueue(),
        this->DeviceFoundPasswordsAddress,
        CL_TRUE /* blocking write */,
        0 /* offset */,
        this->maxFoundPlainLength * this->activeHashesProcessed.size() /* bytes to copy */,
        (void *)this->HostFoundPasswordsAddress,
        NULL, NULL, NULL /* event list stuff */);
    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }
}

void MFNHashTypePlainOpenCL::outputFoundHashes() {
    trace_printf("MFNHashTypePlainOpenCL::outputFoundHashes()\n");
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


void MFNHashTypePlainOpenCL::copySaltArraysToDevice() {
    trace_printf("MFNHashTypeSaltedOpenCL::copySaltArraysToDevice()\n");
        
    cl_int errorCode;
    
    // Device salt lengths.
    errorCode = clEnqueueWriteBuffer (this->OpenCL->getCommandQueue(),
            this->DeviceSaltLengthsAddress,
            CL_TRUE /* blocking write */,
            0 /* offset */,
            this->saltLengths.size() * sizeof(uint32_t) /* bytes to copy */,
            (void *)&this->saltLengths[0],
            NULL, NULL, NULL /* event list stuff */);
    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    // Device salt values.
    errorCode = clEnqueueWriteBuffer (this->OpenCL->getCommandQueue(),
            this->DeviceSaltValuesAddress,
            CL_TRUE /* blocking write */,
            0 /* offset */,
            this->activeSaltsDeviceformat.size() * sizeof(uint32_t) /* bytes to copy */,
            (void *)&this->activeSaltsDeviceformat[0],
            NULL, NULL, NULL /* event list stuff */);
    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }
    
    // Copy the iterations if needed
    if (this->hashAttributes.hashUsesIterationCount) {
        errorCode = clEnqueueWriteBuffer (this->OpenCL->getCommandQueue(),
                this->DeviceIterationCountAddresses,
                CL_TRUE /* blocking write */,
                0 /* offset */,
                this->activeIterationCounts.size() * sizeof(uint32_t),
                (void *)&this->activeIterationCounts[0],
                NULL, NULL, NULL /* event list stuff */);
        if (errorCode != CL_SUCCESS) {
            printf("Error: %s\n", print_cl_errstring(errorCode));
            exit(1);
        }
    }

    numberSaltsCopiedToDevice = this->activeSalts.size();
    this->numberUniqueSalts = numberSaltsCopiedToDevice;
    
    // Push the updated numbers into constant.
    this->copySaltConstantsToDevice();
}