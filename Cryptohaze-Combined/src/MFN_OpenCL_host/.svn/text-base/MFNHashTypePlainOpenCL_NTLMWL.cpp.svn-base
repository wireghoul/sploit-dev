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

#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL_NTLMWL.h"
#include "GRT_OpenCL_host/GRTCLUtils.h"
#include "MFN_Common/MFNDebugging.h"
#include "MFN_Common/MFNDefines.h"
#include "MFN_OpenCL_host/MFNOpenCLMetaprograms.h"
#include "MFN_Common/MFNCommandLineData.h"
#include "CH_Common/CHCharsetNew.h"
#include "OpenCL_Common/GRTOpenCL.h"

// Ugly hack for now.  If true, pulls the kernel into the binary.
#define RELEASE_KERNEL 1

#if RELEASE_KERNEL
extern char MFNHashTypePlainOpenCL_NTLMWL_source[];
#endif


MFNHashTypePlainOpenCL_NTLMWL::MFNHashTypePlainOpenCL_NTLMWL() :  MFNHashTypePlainOpenCL(16) {
    trace_printf("MFNHashTypePlainOpenCL_NTLMWL::MFNHashTypePlainOpenCL_NTLMWL()\n");
    
    this->hashAttributes.hashWordWidth32 = 1;
    this->hashAttributes.hashUsesWordlist = 1;
}

void MFNHashTypePlainOpenCL_NTLMWL::launchKernel() {
    trace_printf("MFNHashTypePlainOpenCL_NTLMWL::launchKernel()\n");
    cl_event kernelLaunchEvent;
    cl_int errorCode;
    size_t numberWorkgroups;
    size_t numberWorkitems;

    numberWorkgroups = this->GPUBlocks * this->GPUThreads;
    numberWorkitems = this->GPUThreads;

    
    klaunch_printf("T %d: Platform/Device: %d/%d\n", this->threadId, this->openCLPlatformId, this->gpuDeviceId);
    klaunch_printf("T %d: Workgroups/Workitems: %d/%d\n", this->threadId, numberWorkgroups, numberWorkitems);

    // Copy the per-step value to the kernel
    errorCode = clSetKernelArg (this->getKernelToRun(), 14, sizeof(cl_uint), &this->perStep);
    errorCode |= clSetKernelArg (this->getKernelToRun(), 20, sizeof(cl_uint), &this->startStep);
    if (errorCode != CL_SUCCESS) {
        printf("Error 1: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    errorCode = clEnqueueNDRangeKernel(this->OpenCL->getCommandQueue(),
            this->getKernelToRun(),
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


void MFNHashTypePlainOpenCL_NTLMWL::copyConstantDataToDevice() {
    trace_printf("MFNHashTypePlainOpenCL_NTLMWL::copyConstantDataToDevice()\n");

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
        errorCode |= clSetKernelArg (currentKernel,  0, sizeof(cl_mem), &this->DeviceForwardCharsetAddress);
        errorCode |= clSetKernelArg (currentKernel,  1, sizeof(cl_mem), &this->DeviceReverseCharsetAddress);
        errorCode |= clSetKernelArg (currentKernel,  2, sizeof(cl_mem), &this->DeviceCharsetLengthsAddress);
        if (this->sharedBitmapSize == 8) {
            errorCode |= clSetKernelArg (currentKernel,  3, sizeof(cl_mem), &this->DeviceBitmap8kb_Address);
        } else if (this->sharedBitmapSize == 16) {
            errorCode |= clSetKernelArg (currentKernel,  3, sizeof(cl_mem), &this->DeviceBitmap16kb_Address);
        } else if (this->sharedBitmapSize == 32) {
            errorCode |= clSetKernelArg (currentKernel,  3, sizeof(cl_mem), &this->DeviceBitmap32kb_Address);
        } else {
            printf("Error: Invalid shared bitmap size!  Must be 8, 16, 32\n");
            exit(1);
        }

        errorCode |= clSetKernelArg (currentKernel,  4, sizeof(cl_ulong), &localNumberHashes);
        errorCode |= clSetKernelArg (currentKernel,  5, sizeof(cl_mem), &this->DeviceHashlistAddress);
        errorCode |= clSetKernelArg (currentKernel,  6, sizeof(cl_mem), &this->DeviceFoundPasswordsAddress);
        errorCode |= clSetKernelArg (currentKernel,  7, sizeof(cl_mem), &this->DeviceSuccessAddress);

        errorCode |= clSetKernelArg (currentKernel,  8, sizeof(cl_mem), &this->DeviceBitmap128mb_a_Address);
        errorCode |= clSetKernelArg (currentKernel,  9, sizeof(cl_mem), &this->DeviceBitmap128mb_b_Address);
        errorCode |= clSetKernelArg (currentKernel, 10, sizeof(cl_mem), &this->DeviceBitmap128mb_c_Address);
        errorCode |= clSetKernelArg (currentKernel, 11, sizeof(cl_mem), &this->DeviceBitmap128mb_d_Address);

        errorCode |= clSetKernelArg (currentKernel, 12, sizeof(cl_mem), &this->DeviceStartPointAddress);
        errorCode |= clSetKernelArg (currentKernel, 13, sizeof(cl_ulong), &localNumberThreads);

        errorCode |= clSetKernelArg (currentKernel, 15, sizeof(cl_mem), &this->DeviceStartPasswords32Address);
        errorCode |= clSetKernelArg (currentKernel, 16, sizeof(cl_mem), &this->DeviceBitmap256kb_a_Address);

        // Wordlist
        errorCode |= clSetKernelArg (currentKernel, 17, sizeof(cl_mem), &this->DeviceWordlistLengths);
        errorCode |= clSetKernelArg (currentKernel, 18, sizeof(cl_mem), &this->DeviceWordlistBlocks);

        if (errorCode != CL_SUCCESS) {
            printf("Thread %d, dev %d: OpenCL error 5: %s. Exiting.\n",
                    this->threadId, this->gpuDeviceId, print_cl_errstring(errorCode));
            exit(1);
        }    
    }
}

void MFNHashTypePlainOpenCL_NTLMWL::copyWordlistSizeToDevice(cl_uint wordCount, cl_uchar blocksPerWord) {
    cl_int errorCode;

    errorCode = clSetKernelArg (this->getKernelToRun(), 19, sizeof(cl_uint), &wordCount);
    errorCode |= clSetKernelArg (this->getKernelToRun(), 21, sizeof(cl_uchar), &blocksPerWord);
    
    if (errorCode != CL_SUCCESS) {
        printf("Error 1: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }
}


std::vector<std::string> MFNHashTypePlainOpenCL_NTLMWL::getHashFileNames() {
    trace_printf("MFNHashTypePlainOpenCL_NTLMWL::getHashFileNames()\n");
    
    std::vector<std::string> returnHashFilenames;
    std::string hashFilename;
    
#if !RELEASE_KERNEL    
    hashFilename = "./src/MFN_OpenCL_device/MFNHashTypePlainOpenCL_NTLMWL.cl";
    returnHashFilenames.push_back(hashFilename);
#endif
    return returnHashFilenames;
}

std::string MFNHashTypePlainOpenCL_NTLMWL::getKernelSourceString() {
    std::string ReturnString;
    
#if RELEASE_KERNEL
    ReturnString += std::string(MFNHashTypePlainOpenCL_NTLMWL_source);
#endif
    
    return ReturnString;
}

std::vector<std::string> 
    MFNHashTypePlainOpenCL_NTLMWL::getHashKernelNamesVector() {
    trace_printf("MFNHashTypePlainOpenCL_NTLMWL::getHashKernelNamesVector()\n");
    
    std::vector<std::string> kernelNames;
    
    // Kernel 0: Block length 1-4
    kernelNames.push_back(std::string("MFNHashTypePlainOpenCL_NTLMWL_B1_4"));

    // Kernel 1: Block length 5-7
    kernelNames.push_back(std::string("MFNHashTypePlainOpenCL_NTLMWL_B5_7"));

    // Kernel 2: Block length 8
    kernelNames.push_back(std::string("MFNHashTypePlainOpenCL_NTLMWL_B8"));

    // Kernel 3: Block length 9-15
    kernelNames.push_back(std::string("MFNHashTypePlainOpenCL_NTLMWL_B9_15"));

    // Kernel 4: Block length 16
    kernelNames.push_back(std::string("MFNHashTypePlainOpenCL_NTLMWL_B16"));

    // Kernel 5: Block length 17-23
    kernelNames.push_back(std::string("MFNHashTypePlainOpenCL_NTLMWL_B17_23"));

    // Kernel 6: Block length 24
    kernelNames.push_back(std::string("MFNHashTypePlainOpenCL_NTLMWL_B24"));

    // Kernel 7: Block length 17-23
    kernelNames.push_back(std::string("MFNHashTypePlainOpenCL_NTLMWL_B25_31"));

    // Kernel 8: Block length 24
    kernelNames.push_back(std::string("MFNHashTypePlainOpenCL_NTLMWL_B32"));

    return kernelNames;
}

cl_kernel MFNHashTypePlainOpenCL_NTLMWL::getKernelToRun() {
    // Pick the kernel based on the block length.
    switch (this->wordlistBlockLength) {
        case 1:
        case 2:
        case 3:
        case 4:
            // MFNHashTypePlainOpenCL_NTLMWL_B1_4
            return this->HashKernelVector[0];
            break;
        case 5:
        case 6:
        case 7:
            // MFNHashTypePlainOpenCL_NTLMWL_B5_7
            return this->HashKernelVector[1];
            break;
        case 8:
            // MFNHashTypePlainOpenCL_NTLMWL_B8
            return this->HashKernelVector[2];
            break;
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
            return this->HashKernelVector[3];
            break;
        case 16:
            return this->HashKernelVector[4];
            break;
        case 17:
        case 18:
        case 19:
        case 20:
        case 21:
        case 22:
        case 23:
            return this->HashKernelVector[5];
            break;
        case 24:
            return this->HashKernelVector[6];
            break;
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
            return this->HashKernelVector[7];
            break;
        case 32:
            return this->HashKernelVector[8];
            break;
        default:
            printf("NO KERNEL SUPPORT YET!\n");
            exit(1);
    }
}


std::string MFNHashTypePlainOpenCL_NTLMWL::getDefineStrings() {
    std::string additionalDefines;
    MFNOpenCLMetaprograms DefineGen;
    int numberBitmaps = 0;
    
    int bitmapAValid = 0;
    int bitmapBValid = 0;
    int bitmapCValid = 0;
    int bitmapDValid = 0;
    
    // Going forward, the 256kb bitmap is ALWAYS used.
    // It's faster this way.
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

    /**
     * Generate defines for various kernel lengths.  These should copy data
     * directly from the source to the destination, as the data may be longer
     * than the single password block can handle.
     */
    
    // Generate the default b0,b1,b2... based password copy.
    additionalDefines += 
            DefineGen.makeBitmapLookupEarlyOut(this->VectorWidth,  "CheckPassword128LE",
            'a', bitmapAValid, 
            'd', bitmapDValid, "",
            'c', bitmapCValid, "",
            'b', bitmapBValid, ";",
            use256kbBitmap);

    // Generate a password copy for block lengths 1-4
    additionalDefines += 
            DefineGen.makeBitmapLookupEarlyOut(this->VectorWidth,
            "CheckWordlistPassword128LE",
            'a', bitmapAValid, 
            'd', bitmapDValid, "",
            'c', bitmapCValid, "",
            'b', bitmapBValid, ";",
            use256kbBitmap, "OpenCLPasswordCheckWordlist128");

    return additionalDefines;
}