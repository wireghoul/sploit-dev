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

//#define KERNEL_LAUNCH_PRINTF 1
//#define TRACE_PRINTF 1

#include "MFN_OpenCL_host/MFNHashTypeSaltedOpenCL_PhpassWL.h"
#include "GRT_OpenCL_host/GRTCLUtils.h"
#include "MFN_Common/MFNDebugging.h"
#include "MFN_Common/MFNDefines.h"
#include "MFN_OpenCL_host/MFNOpenCLMetaprograms.h"
#include "MFN_Common/MFNCommandLineData.h"
#include "CH_Common/CHCharsetNew.h"

#define RELEASE_KERNEL 1

#if RELEASE_KERNEL
extern char MFN_OpenCL_Common_source[];
extern char MFN_OpenCL_MD5_source[];
extern char MFN_OpenCL_PasswordCopiers_source[];
extern char MFNHashTypeSaltedOpenCL_PhpassWL_source[];
#endif


MFNHashTypeSaltedOpenCL_PhpassWL::MFNHashTypeSaltedOpenCL_PhpassWL() :  MFNHashTypePlainOpenCL(16) {
    trace_printf("MFNHashTypeSaltedOpenCL_PhpassWL::MFNHashTypeSaltedOpenCL_PhpassWL()\n");
    this->hashAttributes.hashWordWidth32 = 1;
    this->hashAttributes.hashUsesSalt = 1;
    this->hashAttributes.hashUsesWordlist = 1;
    this->hashAttributes.hashUsesIterationCount = 1;
    // Need 16 bytes of temp space for each hash for iterating.
    this->hashAttributes.hashTempSpaceBytesPerElement = 16;
}

void MFNHashTypeSaltedOpenCL_PhpassWL::launchKernel() {
    trace_printf("MFNHashTypeSaltedOpenCL_PhpassWL::launchKernel()\n");
    cl_event kernelLaunchEvent;
    cl_int errorCode;
    size_t numberWorkgroups;
    size_t numberWorkitems;
    cl_kernel currentKernel;

    numberWorkgroups = this->GPUBlocks * this->GPUThreads;
    numberWorkitems = this->GPUThreads;
    
    currentKernel = this->getKernelToRun();

    
    klaunch_printf("T %d: Platform/Device: %d/%d\n", this->threadId, this->openCLPlatformId, this->gpuDeviceId);
    klaunch_printf("T %d: Workgroups/Workitems: %d/%d\n", this->threadId, numberWorkgroups, numberWorkitems);

    // Copy the per-step value to the kernel
    errorCode = clSetKernelArg (currentKernel, 11, sizeof(cl_uint), &this->perStep);
    errorCode |= clSetKernelArg (currentKernel, 15, sizeof(cl_uint), &this->saltStartOffset);
    errorCode |= clSetKernelArg (currentKernel, 17, sizeof(cl_uint), &this->iterationStartOffset);
    errorCode |= clSetKernelArg (currentKernel, 23, sizeof(cl_uint), &this->startStep);

    if (errorCode != CL_SUCCESS) {
        printf("Error 1: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    errorCode = clEnqueueNDRangeKernel(this->OpenCL->getCommandQueue(),
            currentKernel,
            1 /* numDims */,
            NULL /* offset */,
            &numberWorkgroups,
            &numberWorkitems,
            NULL, NULL,
            &kernelLaunchEvent);
    
    if (errorCode != CL_SUCCESS) {
        printf("Error 2: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    if (clWaitForEvents(1, &kernelLaunchEvent) != CL_SUCCESS) {
        printf("\nError on wait for event!\n");
        fflush(stdout);
    };
    // Release the event to prevent memory leaks.
    clReleaseEvent(kernelLaunchEvent);
}

void MFNHashTypeSaltedOpenCL_PhpassWL::printLaunchDebugData() {
//    printf("Debug data for kernel launch: Thread %d, CUDA Device %d\n", this->threadId, this->gpuDeviceId);
//
//    printf("Host value passwordLengthPlainMD5: %d\n", this->passwordLength);
//    printf("Host value numberOfHashesPlainMD5: %lu\n", this->activeHashesProcessed.size());
//    printf("Host value deviceGlobalHashlistAddressPlainMD5: 0x%16x\n", this->DeviceHashlistAddress);
//    printf("Host value deviceGlobalBitmapAPlainMD5: 0x%16x\n", this->DeviceBitmap128mb_a_Address);
//    printf("Host value deviceGlobalBitmapBPlainMD5: 0x%16x\n", this->DeviceBitmap128mb_b_Address);
//    printf("Host value deviceGlobalBitmapCPlainMD5: 0x%16x\n", this->DeviceBitmap128mb_c_Address);
//    printf("Host value deviceGlobalBitmapDPlainMD5: 0x%16x\n", this->DeviceBitmap128mb_d_Address);
//    printf("Host value deviceGlobalFoundPasswordsPlainMD5: 0x%16x\n", this->DeviceFoundPasswordsAddress);
//    printf("Host value deviceGlobalFoundPasswordFlagsPlainMD5: 0x%16x\n", this->DeviceSuccessAddress);
//    printf("Host value deviceGlobalStartPointsPlainMD5: 0x%16x\n", this->DeviceStartPointAddress);
}

void MFNHashTypeSaltedOpenCL_PhpassWL::copyConstantDataToDevice() {
    trace_printf("MFNHashTypeSaltedOpenCL_PhpassWL::copyConstantDataToDevice()\n");

    cl_int errorCode;
    
    // Begin copying constant data to the device.
    errorCode = clEnqueueWriteBuffer (this->OpenCL->getCommandQueue(),
            this->DeviceForwardCharsetAddress,
            CL_TRUE /* blocking write */,
            0 /* offset */,
            this->charsetForwardLookup.size() /* bytes to copy */,
            (void *)&this->charsetForwardLookup[0],
            NULL, NULL, NULL /* event list stuff */);
    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    errorCode = clEnqueueWriteBuffer (this->OpenCL->getCommandQueue(),
            this->DeviceReverseCharsetAddress,
            CL_TRUE /* blocking write */,
            0 /* offset */,
            this->charsetReverseLookup.size() /* bytes to copy */,
            (void *)&this->charsetReverseLookup[0],
            NULL, NULL, NULL /* event list stuff */);
    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    errorCode = clEnqueueWriteBuffer (this->OpenCL->getCommandQueue(),
            this->DeviceCharsetLengthsAddress,
            CL_TRUE /* blocking write */,
            0 /* offset */,
            this->charsetLengths.size() /* bytes to copy */,
            (void *)&this->charsetLengths[0],
            NULL, NULL, NULL /* event list stuff */);
    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    errorCode = clEnqueueWriteBuffer (this->OpenCL->getCommandQueue(),
            this->DeviceBitmap8kb_Address,
            CL_TRUE /* blocking write */,
            0 /* offset */,
            8192 /* bytes to copy */,
            (void *)&this->sharedBitmap8kb_a[0],
            NULL, NULL, NULL /* event list stuff */);
    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }
    
    errorCode = clEnqueueWriteBuffer (this->OpenCL->getCommandQueue(),
            this->DeviceBitmap16kb_Address,
            CL_TRUE /* blocking write */,
            0 /* offset */,
            16384 /* bytes to copy */,
            (void *)&this->sharedBitmap16kb_a[0],
            NULL, NULL, NULL /* event list stuff */);
    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    errorCode = clEnqueueWriteBuffer (this->OpenCL->getCommandQueue(),
            this->DeviceBitmap32kb_Address,
            CL_TRUE /* blocking write */,
            0 /* offset */,
            32768 /* bytes to copy */,
            (void *)&this->sharedBitmap32kb_a[0],
            NULL, NULL, NULL /* event list stuff */);
    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }
    
    errorCode = clEnqueueWriteBuffer (this->OpenCL->getCommandQueue(),
            this->DeviceBitmap256kb_a_Address,
            CL_TRUE /* blocking write */,
            0 /* offset */,
            256*1024 /* bytes to copy */,
            (void *)&this->globalBitmap256kb_a[0],
            NULL, NULL, NULL /* event list stuff */);
    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }
    
    // Copy the values into a variable that can be accessed as a pointer.
    uint64_t localNumberHashes = this->activeHashesProcessed.size();
    uint64_t localNumberThreads = this->TotalKernelWidth;
        // Need to set kernel arguments for all the kernels present in this environment.
    for (int i = 0; i < this->HashKernelVector.size(); i++) {
        cl_kernel currentKernel = this->HashKernelVector[i];

        errorCode = 0;
        if (this->sharedBitmapSize == 8) {
            errorCode |= clSetKernelArg (currentKernel,  0, sizeof(cl_mem), &this->DeviceBitmap8kb_Address);
        } else if (this->sharedBitmapSize == 16) {
            errorCode |= clSetKernelArg (currentKernel,  0, sizeof(cl_mem), &this->DeviceBitmap16kb_Address);
        } else if (this->sharedBitmapSize == 32) {
            errorCode |= clSetKernelArg (currentKernel,  0, sizeof(cl_mem), &this->DeviceBitmap32kb_Address);
        } else {
            printf("Error: Invalid shared bitmap size!  Must be 8, 16, 32\n");
            exit(1);
        }

        errorCode |= clSetKernelArg (currentKernel,  1, sizeof(cl_mem), &this->DeviceBitmap256kb_a_Address);

        errorCode |= clSetKernelArg (currentKernel,  2, sizeof(cl_mem), &this->DeviceBitmap128mb_a_Address);
        errorCode |= clSetKernelArg (currentKernel,  3, sizeof(cl_mem), &this->DeviceBitmap128mb_b_Address);
        errorCode |= clSetKernelArg (currentKernel,  4, sizeof(cl_mem), &this->DeviceBitmap128mb_c_Address);
        errorCode |= clSetKernelArg (currentKernel,  5, sizeof(cl_mem), &this->DeviceBitmap128mb_d_Address);

        
        errorCode |= clSetKernelArg (currentKernel,  6, sizeof(cl_ulong), &localNumberHashes);
        errorCode |= clSetKernelArg (currentKernel,  7, sizeof(cl_mem), &this->DeviceHashlistAddress);
        errorCode |= clSetKernelArg (currentKernel,  8, sizeof(cl_mem), &this->DeviceFoundPasswordsAddress);
        errorCode |= clSetKernelArg (currentKernel,  9, sizeof(cl_mem), &this->DeviceSuccessAddress);

        errorCode |= clSetKernelArg (currentKernel, 10, sizeof(cl_ulong), &localNumberThreads);
        
        // Salt constants in a separate function

        errorCode |= clSetKernelArg (currentKernel, 16, sizeof(cl_mem), &this->DeviceIterationCountAddresses);
        errorCode |= clSetKernelArg (currentKernel, 18, sizeof(cl_mem), &this->DeviceTempSpaceAddress);

        // Wordlist address
        errorCode |= clSetKernelArg (currentKernel, 19, sizeof(cl_mem), &this->DeviceWordlistLengths);
        errorCode |= clSetKernelArg (currentKernel, 20, sizeof(cl_mem), &this->DeviceWordlistBlocks);

        if (errorCode != CL_SUCCESS) {
            printf("Thread %d, dev %d: OpenCL error 5: %s. Exiting.\n",
                    this->threadId, this->gpuDeviceId, print_cl_errstring(errorCode));
            exit(1);
        }    
    }
    
    if (errorCode != CL_SUCCESS) {
        printf("Thread %d, dev %d: OpenCL error 5: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, print_cl_errstring(errorCode));
        exit(1);
    }
}

void MFNHashTypeSaltedOpenCL_PhpassWL::copyWordlistSizeToDevice(cl_uint wordCount, cl_uchar blocksPerWord) {
    trace_printf("MFNHashTypeSaltedOpenCL_PhpassWL::copyWordlistSizeToDevice()\n");
    
    cl_int errorCode;

    errorCode = clSetKernelArg (this->getKernelToRun(), 21, sizeof(cl_uint), &wordCount);
    errorCode |= clSetKernelArg (this->getKernelToRun(), 22, sizeof(cl_uchar), &blocksPerWord);
    
    if (errorCode != CL_SUCCESS) {
        printf("Error 1: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }
}


void MFNHashTypeSaltedOpenCL_PhpassWL::copySaltConstantsToDevice() {
    trace_printf("MFNHashTypeSaltedOpenCL_PhpassWL::copySaltConstantsToDevice()\n");
    
    cl_int errorCode = 0;
    
    // Salted hash data
    uint64_t localNumberSaltValues = this->numberSaltsCopiedToDevice;
    for (int i = 0; i < this->HashKernelVector.size(); i++) {
        cl_kernel currentKernel = this->HashKernelVector[i];
        errorCode |= clSetKernelArg (currentKernel, 12, sizeof(cl_ulong), &localNumberSaltValues);
        errorCode |= clSetKernelArg (currentKernel, 13, sizeof(cl_mem), &this->DeviceSaltLengthsAddress);
        errorCode |= clSetKernelArg (currentKernel, 14, sizeof(cl_mem), &this->DeviceSaltValuesAddress);
    }

    if (errorCode != CL_SUCCESS) {
        printf("Thread %d, dev %d: OpenCL error 5: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, print_cl_errstring(errorCode));
        exit(1);
    }
}

std::string MFNHashTypeSaltedOpenCL_PhpassWL::getKernelSourceString() {
    trace_printf("MFNHashTypeSaltedOpenCL_PhpassWL::getKernelSourceString()\n");
    
    std::string ReturnString;
    
#if RELEASE_KERNEL
    ReturnString += std::string(MFN_OpenCL_Common_source);
    ReturnString += std::string(MFN_OpenCL_MD5_source);
    ReturnString += std::string(MFN_OpenCL_PasswordCopiers_source);
    ReturnString += std::string(MFNHashTypeSaltedOpenCL_PhpassWL_source);
#endif
    
    return ReturnString;
}

std::vector<std::string> MFNHashTypeSaltedOpenCL_PhpassWL::getHashFileNames() {
    trace_printf("MFNHashTypeSaltedOpenCL_PhpassWL::getHashFileNames()\n");
    
    std::vector<std::string> returnHashFilenames;
    std::string hashFilename;
    
#if !RELEASE_KERNEL
    hashFilename = "./src/MFN_OpenCL_device/MFN_OpenCL_Common.cl";
    returnHashFilenames.push_back(hashFilename);
    hashFilename = "./src/MFN_OpenCL_device/MFN_OpenCL_MD5.cl";
    returnHashFilenames.push_back(hashFilename);
    hashFilename = "./src/MFN_OpenCL_device/MFNHashTypeSaltedOpenCL_PhpassWL.cl";
    returnHashFilenames.push_back(hashFilename);
#endif
    return returnHashFilenames;
}


std::vector<std::string> 
    MFNHashTypeSaltedOpenCL_PhpassWL::getHashKernelNamesVector() {
    trace_printf("MFNHashTypePlainOpenCL_MD5WL::getHashKernelNamesVector()\n");
    
    std::vector<std::string> kernelNames;
    
    // Kernel 0: Block length 1-4
    kernelNames.push_back(std::string("MFNHashTypeSaltedOpenCL_PhpassWL_B1_4"));

    // Kernel 1: Block length 5-10
    kernelNames.push_back(std::string("MFNHashTypeSaltedOpenCL_PhpassWL_B5_10"));

    // Kernel 2: Block length 11-12
    kernelNames.push_back(std::string("MFNHashTypeSaltedOpenCL_PhpassWL_B11_12"));

    // Kernel 3: Block length 13-14
    kernelNames.push_back(std::string("MFNHashTypeSaltedOpenCL_PhpassWL_B13_14"));

    // Kernel 4: Block length 15-26
    kernelNames.push_back(std::string("MFNHashTypeSaltedOpenCL_PhpassWL_B15_26"));

    // Kernel 5: Block length 27-28
    kernelNames.push_back(std::string("MFNHashTypeSaltedOpenCL_PhpassWL_B27_28"));

    // Kernel 6: Block length 29-30
    kernelNames.push_back(std::string("MFNHashTypeSaltedOpenCL_PhpassWL_B29_30"));

    // Kernel 7: Block length 31-32
    kernelNames.push_back(std::string("MFNHashTypeSaltedOpenCL_PhpassWL_B31_32"));

    return kernelNames;
}

cl_kernel MFNHashTypeSaltedOpenCL_PhpassWL::getKernelToRun() {
    trace_printf("MFNHashTypeSaltedOpenCL_PhpassWL::getKernelToRun()\n");
    // Pick the kernel based on the block length.
    klaunch_printf("Choosing kernel for block length %d\n", this->wordlistBlockLength);
    switch (this->wordlistBlockLength) {
        case 1:
        case 2:
        case 3:
        case 4:
            return this->HashKernelVector[0];
            break;
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
            return this->HashKernelVector[1];
            break;
        case 11:
        case 12:
            return this->HashKernelVector[2];
            break;
        case 13:
        case 14:
            return this->HashKernelVector[3];
            break;
        case 15:
        case 16:
        case 17:
        case 18:
        case 19:
        case 20:
        case 21:
        case 22:
        case 23:
        case 24:
        case 25:
        case 26:
            return this->HashKernelVector[4];
            break;
        case 27:
        case 28:
            return this->HashKernelVector[5];
            break;
        case 29:
        case 30:
            return this->HashKernelVector[6];
            break;
        case 31:
        case 32:
            return this->HashKernelVector[7];
            break;
        default:
            printf("NO KERNEL SUPPORT YET!\n");
            exit(1);
    }
}


std::string MFNHashTypeSaltedOpenCL_PhpassWL::getDefineStrings() {
    trace_printf("MFNHashTypeSaltedOpenCL_PhpassWL::getDefineStrings()\n");
    
    std::string additionalDefines;
    MFNOpenCLMetaprograms DefineGen;
    int numberBitmaps = 0;
    
    int bitmapAValid = 0;
    int bitmapBValid = 0;
    int bitmapCValid = 0;
    int bitmapDValid = 0;
    
    // Build the lookup bitmap
    if (this->DeviceBitmap128mb_a_Address != 0) {
        numberBitmaps++;
        bitmapAValid = 1;
    }
    if (this->DeviceBitmap128mb_b_Address != 0) {
        numberBitmaps++;
        bitmapBValid = 1;
    }
    if (this->DeviceBitmap128mb_c_Address != 0) {
        numberBitmaps++;
        bitmapCValid = 1;
    }
    if (this->DeviceBitmap128mb_d_Address != 0) {
        numberBitmaps++;
        bitmapDValid = 1;
    }

    klaunch_printf("numberBitmaps: %d\n", numberBitmaps);

    additionalDefines += 
            DefineGen.makeBitmapLookupEarlyOut(this->VectorWidth,
            "CheckWordlistPassword128LE",
            'a', bitmapAValid, 
            'd', bitmapDValid, "",
            'c', bitmapCValid, "",
            'b', bitmapBValid, ";",
            1, "OpenCLPasswordCheckWordlist128");
        
    return additionalDefines;
}