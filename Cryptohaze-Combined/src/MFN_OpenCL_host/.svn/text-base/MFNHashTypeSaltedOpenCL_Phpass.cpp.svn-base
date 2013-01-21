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

#include "MFN_OpenCL_host/MFNHashTypeSaltedOpenCL_Phpass.h"
#include "GRT_OpenCL_host/GRTCLUtils.h"
#include "MFN_Common/MFNDebugging.h"
#include "MFN_Common/MFNDefines.h"
#include "MFN_OpenCL_host/MFNOpenCLMetaprograms.h"
#include "MFN_Common/MFNCommandLineData.h"
#include "CH_Common/CHCharsetNew.h"

// Ugly hack for now.  If true, pulls the kernel into the binary.
#define RELEASE_KERNEL 1

#if RELEASE_KERNEL
extern char MFN_OpenCL_Common_source[];
extern char MFN_OpenCL_MD5_source[];
extern char MFN_OpenCL_PasswordCopiers_source[];
extern char MFNHashTypeSaltedOpenCL_Phpass_source[];
#endif


MFNHashTypeSaltedOpenCL_Phpass::MFNHashTypeSaltedOpenCL_Phpass() :  MFNHashTypePlainOpenCL(16) {
    trace_printf("MFNHashTypeSaltedOpenCL_Phpass::MFNHashTypeSaltedOpenCL_Phpass()\n");
    this->hashAttributes.hashWordWidth32 = 1;
    this->hashAttributes.hashUsesSalt = 1;
    this->hashAttributes.hashUsesIterationCount = 1;
    // Need 16 bytes of temp space for each hash for iterating.
    this->hashAttributes.hashTempSpaceBytesPerElement = 16;
}

void MFNHashTypeSaltedOpenCL_Phpass::launchKernel() {
    trace_printf("MFNHashTypeSaltedOpenCL_Phpass::launchKernel()\n");
    cl_event kernelLaunchEvent;
    cl_int errorCode;
    size_t numberWorkgroups;
    size_t numberWorkitems;

    numberWorkgroups = this->GPUBlocks * this->GPUThreads;
    numberWorkitems = this->GPUThreads;

    
    klaunch_printf("T %d: Platform/Device: %d/%d\n", this->threadId, this->openCLPlatformId, this->gpuDeviceId);
    klaunch_printf("T %d: Workgroups/Workitems: %d/%d\n", this->threadId, numberWorkgroups, numberWorkitems);

    // Copy the per-step value to the kernel
    errorCode = clSetKernelArg (this->HashKernel, 12, sizeof(cl_uint), &this->perStep);
    errorCode |= clSetKernelArg (this->HashKernel, 16, sizeof(cl_uint), &this->saltStartOffset);
    errorCode |= clSetKernelArg (this->HashKernel, 18, sizeof(cl_uint), &this->iterationStartOffset);
    
    if (errorCode != CL_SUCCESS) {
        printf("Error 1: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    errorCode = clEnqueueNDRangeKernel(this->OpenCL->getCommandQueue(),
            this->HashKernel,
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

void MFNHashTypeSaltedOpenCL_Phpass::copyConstantDataToDevice() {
    trace_printf("MFNHashTypeSaltedOpenCL_Phpass::copyConstantDataToDevice()\n");

    cl_int errorCode;

    
    // Begin copying constant data to the device.
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
    
    errorCode = 0;
    if (this->sharedBitmapSize == 8) {
        errorCode |= clSetKernelArg (this->HashKernel,  0, sizeof(cl_mem), &this->DeviceBitmap8kb_Address);
    } else if (this->sharedBitmapSize == 16) {
        errorCode |= clSetKernelArg (this->HashKernel,  0, sizeof(cl_mem), &this->DeviceBitmap16kb_Address);
    } else if (this->sharedBitmapSize == 32) {
        errorCode |= clSetKernelArg (this->HashKernel,  0, sizeof(cl_mem), &this->DeviceBitmap32kb_Address);
    } else {
        printf("Error: Invalid shared bitmap size!  Must be 8, 16, 32\n");
        exit(1);
    }
    errorCode |= clSetKernelArg (this->HashKernel, 1, sizeof(cl_mem), &this->DeviceBitmap256kb_a_Address);

    errorCode |= clSetKernelArg (this->HashKernel,  2, sizeof(cl_mem), &this->DeviceBitmap128mb_a_Address);
    errorCode |= clSetKernelArg (this->HashKernel,  3, sizeof(cl_mem), &this->DeviceBitmap128mb_b_Address);
    errorCode |= clSetKernelArg (this->HashKernel,  4, sizeof(cl_mem), &this->DeviceBitmap128mb_c_Address);
    errorCode |= clSetKernelArg (this->HashKernel,  5, sizeof(cl_mem), &this->DeviceBitmap128mb_d_Address);
    
    errorCode |= clSetKernelArg (this->HashKernel,  6, sizeof(cl_ulong), &localNumberHashes);
    errorCode |= clSetKernelArg (this->HashKernel,  7, sizeof(cl_mem), &this->DeviceHashlistAddress);
    errorCode |= clSetKernelArg (this->HashKernel,  8, sizeof(cl_mem), &this->DeviceFoundPasswordsAddress);
    errorCode |= clSetKernelArg (this->HashKernel,  9, sizeof(cl_mem), &this->DeviceSuccessAddress);

    errorCode |= clSetKernelArg (this->HashKernel, 10, sizeof(cl_mem), &this->DeviceStartPasswords32Address);
    errorCode |= clSetKernelArg (this->HashKernel, 11, sizeof(cl_ulong), &localNumberThreads);
    
    errorCode |= clSetKernelArg (this->HashKernel, 17, sizeof(cl_mem), &this->DeviceIterationCountAddresses);

    errorCode |= clSetKernelArg (this->HashKernel, 19, sizeof(cl_mem), &this->DeviceTempSpaceAddress);

    if (errorCode != CL_SUCCESS) {
        printf("Thread %d, dev %d: OpenCL error 5: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, print_cl_errstring(errorCode));
        exit(1);
    }
}

void MFNHashTypeSaltedOpenCL_Phpass::copySaltConstantsToDevice() {
    trace_printf("MFNHashTypeSaltedOpenCL_Phpass::copySaltConstantsToDevice()\n");
    
    cl_int errorCode = 0;
    
    // Salted hash data
    uint64_t localNumberSaltValues = this->numberSaltsCopiedToDevice;

    errorCode |= clSetKernelArg (this->HashKernel, 13, sizeof(cl_ulong), &localNumberSaltValues);
    errorCode |= clSetKernelArg (this->HashKernel, 14, sizeof(cl_mem), &this->DeviceSaltLengthsAddress);
    errorCode |= clSetKernelArg (this->HashKernel, 15, sizeof(cl_mem), &this->DeviceSaltValuesAddress);

    if (errorCode != CL_SUCCESS) {
        printf("Thread %d, dev %d: OpenCL error 5: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, print_cl_errstring(errorCode));
        exit(1);
    }
}

std::string MFNHashTypeSaltedOpenCL_Phpass::getKernelSourceString() {
    std::string ReturnString;
    
#if RELEASE_KERNEL
    ReturnString += std::string(MFN_OpenCL_Common_source);
    ReturnString += std::string(MFN_OpenCL_MD5_source);
    ReturnString += std::string(MFN_OpenCL_PasswordCopiers_source);
    ReturnString += std::string(MFNHashTypeSaltedOpenCL_Phpass_source);
#endif
    
    return ReturnString;
}

std::vector<std::string> MFNHashTypeSaltedOpenCL_Phpass::getHashFileNames() {
    trace_printf("MFNHashTypeSaltedOpenCL_Phpass::getHashFileNames()\n");
    
    std::vector<std::string> returnHashFilenames;
    std::string hashFilename;
    
#if !RELEASE_KERNEL
    hashFilename = "./src/MFN_OpenCL_device/MFN_OpenCL_Common.cl";
    returnHashFilenames.push_back(hashFilename);
    hashFilename = "./src/MFN_OpenCL_device/MFN_OpenCL_MD5.cl";
    returnHashFilenames.push_back(hashFilename);
    hashFilename = "./src/MFN_OpenCL_device/MFNHashTypeSaltedOpenCL_Phpass.cl";
    returnHashFilenames.push_back(hashFilename);
#endif
    return returnHashFilenames;
}
    
std::string MFNHashTypeSaltedOpenCL_Phpass::getHashKernelName() {
    trace_printf("MFNHashTypeSaltedOpenCL_Phpass::getHashKernelName()\n");
    
    std::string hashKernel = "MFNHashTypeSaltedOpenCL_Phpass";
    return hashKernel;
}

std::string MFNHashTypeSaltedOpenCL_Phpass::getDefineStrings() {
    std::string additionalDefines;
    MFNOpenCLMetaprograms DefineGen;
    int numberBitmaps = 0;
    
    int bitmapAValid = 0;
    int bitmapBValid = 0;
    int bitmapCValid = 0;
    int bitmapDValid = 0;
    
    int use256kbBitmap = 1;
    
    
    if (this->isSingleCharset) {
        klaunch_printf("Making single charset incrementors.\n");
        klaunch_printf("passLength: %d\n", this->passwordLength);
        klaunch_printf("vectorWidth: %d\n", this->VectorWidth);
        additionalDefines += 
                DefineGen.makePasswordNoMemSingleIncrementorsLE(
                this->passwordLength, this->VectorWidth, 
                this->Charset->getCharset());
    } else {
        klaunch_printf("Making multiple charset incrementors.\n");
        klaunch_printf("passLength: %d\n", this->passwordLength);
        klaunch_printf("vectorWidth: %d\n", this->VectorWidth);
        additionalDefines += 
                DefineGen.makePasswordNoMemMultipleIncrementorsLE(
                this->passwordLength, this->VectorWidth, 
                this->Charset->getCharset());
    }
    
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
            DefineGen.makeBitmapLookupEarlyOut(this->VectorWidth,  "CheckPassword128LE",
            'a', bitmapAValid, 
            'd', bitmapDValid, "",
            'c', bitmapCValid, "",
            'b', bitmapBValid, "",
            use256kbBitmap,
            std::string("OpenCLPasswordCheck128_EarlyOut"),
            std::string("PHPASS_LOAD_PASSWORDS_FROM_GLOBAL_INCREMENT()"));
        
    return additionalDefines;
}