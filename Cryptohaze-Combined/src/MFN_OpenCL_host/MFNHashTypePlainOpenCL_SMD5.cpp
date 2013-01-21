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

#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL_SMD5.h"
#include "GRT_OpenCL_host/GRTCLUtils.h"
#include "MFN_Common/MFNDebugging.h"
#include "MFN_Common/MFNDefines.h"
#include "MFN_OpenCL_host/MFNOpenCLMetaprograms.h"
#include "MFN_Common/MFNCommandLineData.h"
#include "CH_Common/CHCharsetNew.h"

// Ugly hack for now.  If true, pulls the kernel into the binary.
#define RELEASE_KERNEL 1

#if RELEASE_KERNEL
extern char MFNHashTypePlainOpenCL_SMD5_source[];
#endif


#define MD5ROTATE_RIGHT(x, n) (((x) >> (n)) | ((x) << (32-(n))))

#define REV_II(a,b,c,d,data,shift,constant) \
    a = MD5ROTATE_RIGHT((a - b), shift) - data - constant - (c ^ (b | (~d)));

#define MD5I(x, y, z) ((y) ^ ((x) | (~z)))
#define MD5ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))
#define MD5II(a, b, c, d, x, s, ac) { \
 (a) += MD5I ((b), (c), (d)) + (x) + (uint32_t)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }

#define MD5S41 6
#define MD5S42 10
#define MD5S43 15
#define MD5S44 21

MFNHashTypePlainOpenCL_SMD5::MFNHashTypePlainOpenCL_SMD5() :  MFNHashTypePlainOpenCL(16) {
    trace_printf("MFNHashTypePlainOpenCL_SMD5::MFNHashTypePlainOpenCL_SMD5()\n");
}

void MFNHashTypePlainOpenCL_SMD5::launchKernel() {
    trace_printf("MFNHashTypePlainOpenCL_SMD5::launchKernel()\n");
    cl_event kernelLaunchEvent;
    cl_int errorCode;
    size_t numberWorkgroups;
    size_t numberWorkitems;

    numberWorkgroups = this->GPUBlocks * this->GPUThreads;
    numberWorkitems = this->GPUThreads;

    
    klaunch_printf("T %d: Platform/Device: %d/%d\n", this->threadId, this->openCLPlatformId, this->gpuDeviceId);
    klaunch_printf("T %d: Workgroups/Workitems: %d/%d\n", this->threadId, numberWorkgroups, numberWorkitems);

    // Copy the per-step value to the kernel
    errorCode = clSetKernelArg (this->HashKernel, 4, sizeof(cl_uint), &this->perStep);
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

void MFNHashTypePlainOpenCL_SMD5::printLaunchDebugData() {
//    printf("Debug data for kernel launch: Thread %d, CUDA Device %d\n", this->threadId, this->gpuDeviceId);
//
//    printf("Host value passwordLengthPlainSMD5: %d\n", this->passwordLength);
//    printf("Host value numberOfHashesPlainSMD5: %lu\n", this->activeHashesProcessed.size());
//    printf("Host value deviceGlobalHashlistAddressPlainSMD5: 0x%16x\n", this->DeviceHashlistAddress);
//    printf("Host value deviceGlobalBitmapAPlainSMD5: 0x%16x\n", this->DeviceBitmap128mb_a_Address);
//    printf("Host value deviceGlobalBitmapBPlainSMD5: 0x%16x\n", this->DeviceBitmap128mb_b_Address);
//    printf("Host value deviceGlobalBitmapCPlainSMD5: 0x%16x\n", this->DeviceBitmap128mb_c_Address);
//    printf("Host value deviceGlobalBitmapDPlainSMD5: 0x%16x\n", this->DeviceBitmap128mb_d_Address);
//    printf("Host value deviceGlobalFoundPasswordsPlainSMD5: 0x%16x\n", this->DeviceFoundPasswordsAddress);
//    printf("Host value deviceGlobalFoundPasswordFlagsPlainSMD5: 0x%16x\n", this->DeviceSuccessAddress);
//    printf("Host value deviceGlobalStartPointsPlainSMD5: 0x%16x\n", this->DeviceStartPointAddress);
}

std::vector<uint8_t> MFNHashTypePlainOpenCL_SMD5::preProcessHash(std::vector<uint8_t> rawHash) {
    trace_printf("MFNHashTypePlainOpenCL_SMD5::preProcessHash()\n");
    
    uint32_t a, b, c, d;
    uint32_t *hash32 = (uint32_t *)&rawHash[0];

    /*
    printf("Raw Hash: ");
    for (i = 0; i < rawHash.size(); i++) {
        printf("%02x", rawHash[i]);
    }
    printf("\n");
    */
    a = hash32[0];
    b = hash32[1];
    c = hash32[2];
    d = hash32[3];
    
    a -= 0x67452301;
    b -= 0xefcdab89;
    c -= 0x98badcfe;
    d -= 0x10325476;
    
    if (this->passwordLength < 8) {
        REV_II (b, c, d, a, 0x00 /*b9*/, MD5S44, 0xeb86d391); //64
        REV_II (c, d, a, b, 0x00 /*b2*/, MD5S43, 0x2ad7d2bb); //63
        REV_II (d, a, b, c, 0x00 /*b11*/, MD5S42, 0xbd3af235); //62
        REV_II (a, b, c, d, 0x00 /*b4*/, MD5S41, 0xf7537e82); //61
        REV_II (b, c, d, a, 0x00 /*b13*/, MD5S44, 0x4e0811a1); //60
        REV_II (c, d, a, b, 0x00 /*b6*/, MD5S43, 0xa3014314); //59
        REV_II (d, a, b, c, 0x00 /*b15*/, MD5S42, 0xfe2ce6e0); //58
        REV_II (a, b, c, d, 0x00 /*b8*/, MD5S41, 0x6fa87e4f); //57
    } else if (this->passwordLength == 8) {
        REV_II (b, c, d, a, 0x00 /*b9*/, MD5S44, 0xeb86d391); //64
        // Padding bit will be set
        REV_II (c, d, a, b, 0x00000080 /*b2*/, MD5S43, 0x2ad7d2bb); //63
        REV_II (d, a, b, c, 0x00 /*b11*/, MD5S42, 0xbd3af235); //62
        REV_II (a, b, c, d, 0x00 /*b4*/, MD5S41, 0xf7537e82); //61
        REV_II (b, c, d, a, 0x00 /*b13*/, MD5S44, 0x4e0811a1); //60
        REV_II (c, d, a, b, 0x00 /*b6*/, MD5S43, 0xa3014314); //59
        REV_II (d, a, b, c, 0x00 /*b15*/, MD5S42, 0xfe2ce6e0); //58
        REV_II (a, b, c, d, 0x00 /*b8*/, MD5S41, 0x6fa87e4f); //57
    }
    
    hash32[0] = a;
    hash32[1] = b;
    hash32[2] = c;
    hash32[3] = d;
    /*
    printf("Preprocessed Hash: ");
    for (i = 0; i < rawHash.size(); i++) {
        printf("%02x", rawHash[i]);
    }
    printf("\n");
    
    printf("Returning rawHash\n");
    */
    return rawHash;
}

std::vector<uint8_t> MFNHashTypePlainOpenCL_SMD5::postProcessHash(std::vector<uint8_t> processedHash) {
    trace_printf("MFNHashTypePlainOpenCL_SMD5::postProcessHash()\n");
    
    uint32_t a, b, c, d;
    uint32_t *hash32 = (uint32_t *)&processedHash[0];
    
    a = hash32[0];
    b = hash32[1];
    c = hash32[2];
    d = hash32[3];

    if (this->passwordLength < 8) {
        MD5II(a, b, c, d, 0x00, MD5S41, 0x6fa87e4f); /* 57 */
        MD5II(d, a, b, c, 0x00, MD5S42, 0xfe2ce6e0); /* 58 */
        MD5II(c, d, a, b, 0x00, MD5S43, 0xa3014314); /* 59 */
        MD5II(b, c, d, a, 0x00, MD5S44, 0x4e0811a1); /* 60 */
        MD5II(a, b, c, d, 0x00, MD5S41, 0xf7537e82); /* 61 */
        MD5II(d, a, b, c, 0x00, MD5S42, 0xbd3af235); /* 62 */
        MD5II(c, d, a, b, 0x00, MD5S43, 0x2ad7d2bb); /* 63 */
        MD5II(b, c, d, a, 0x00, MD5S44, 0xeb86d391); /* 64 */
    } else if (this->passwordLength == 8) {
        MD5II(a, b, c, d, 0x00, MD5S41, 0x6fa87e4f); /* 57 */
        MD5II(d, a, b, c, 0x00, MD5S42, 0xfe2ce6e0); /* 58 */
        MD5II(c, d, a, b, 0x00, MD5S43, 0xa3014314); /* 59 */
        MD5II(b, c, d, a, 0x00, MD5S44, 0x4e0811a1); /* 60 */
        MD5II(a, b, c, d, 0x00, MD5S41, 0xf7537e82); /* 61 */
        MD5II(d, a, b, c, 0x00, MD5S42, 0xbd3af235); /* 62 */
        MD5II(c, d, a, b, 0x00000080, MD5S43, 0x2ad7d2bb); /* 63 */
        MD5II(b, c, d, a, 0x00, MD5S44, 0xeb86d391); /* 64 */
    }
    
    a += 0x67452301;
    b += 0xefcdab89;
    c += 0x98badcfe;
    d += 0x10325476;
    
    hash32[0] = a;
    hash32[1] = b;
    hash32[2] = c;
    hash32[3] = d;

    
    return processedHash;
}

void MFNHashTypePlainOpenCL_SMD5::copyConstantDataToDevice() {
    trace_printf("MFNHashTypePlainOpenCL_SMD5::copyConstantDataToDevice()\n");

    cl_int errorCode;

    // Copy the values into a variable that can be accessed as a pointer.
    uint64_t localNumberThreads = this->TotalKernelWidth;
    
    errorCode = 0;
    errorCode |= clSetKernelArg (this->HashKernel,  0, sizeof(cl_mem), &this->DeviceHashlistAddress);
    errorCode |= clSetKernelArg (this->HashKernel,  1, sizeof(cl_mem), &this->DeviceFoundPasswordsAddress);
    errorCode |= clSetKernelArg (this->HashKernel,  2, sizeof(cl_mem), &this->DeviceSuccessAddress);

    errorCode |= clSetKernelArg (this->HashKernel, 3, sizeof(cl_ulong), &localNumberThreads);
    
    errorCode |= clSetKernelArg (this->HashKernel, 5, sizeof(cl_mem), &this->DeviceStartPasswords32Address);
    

    if (errorCode != CL_SUCCESS) {
        printf("Thread %d, dev %d: OpenCL error 5: %s. Exiting.\n",
                this->threadId, this->gpuDeviceId, print_cl_errstring(errorCode));
        exit(1);
    }
}

std::vector<std::string> MFNHashTypePlainOpenCL_SMD5::getHashFileNames() {
    trace_printf("MFNHashTypePlainOpenCL_SMD5::getHashFileNames()\n");
    
    std::vector<std::string> returnHashFilenames;
    std::string hashFilename;
    
#if !RELEASE_KERNEL    
    hashFilename = "./src/MFN_OpenCL_device/MFNHashTypePlainOpenCL_SMD5.cl";
    returnHashFilenames.push_back(hashFilename);
#endif
    return returnHashFilenames;
}

std::string MFNHashTypePlainOpenCL_SMD5::getKernelSourceString() {
    std::string ReturnString;
    
#if RELEASE_KERNEL
    ReturnString += std::string(MFNHashTypePlainOpenCL_SMD5_source);
#endif
    
    return ReturnString;
}

std::string MFNHashTypePlainOpenCL_SMD5::getHashKernelName() {
    trace_printf("MFNHashTypePlainOpenCL_SMD5::getHashKernelName()\n");
    
    std::string hashKernel = "MFNHashTypePlainOpenCL_SMD5";
    return hashKernel;
}

std::string MFNHashTypePlainOpenCL_SMD5::getDefineStrings() {
    std::string additionalDefines;
    MFNOpenCLMetaprograms DefineGen;

    if (this->isSingleCharset) {
        klaunch_printf("Making single charset incrementors.\n");
        klaunch_printf("passLength: %d\n", this->passwordLength);
        klaunch_printf("vectorWidth: %d\n", this->VectorWidth);
        additionalDefines += 
                DefineGen.makePasswordNoMemSingleIncrementorsLE(
                this->passwordLength, this->VectorWidth, 
                this->Charset->getCharset(), 1, 4);
    } else {
        klaunch_printf("Making multiple charset incrementors.\n");
        klaunch_printf("passLength: %d\n", this->passwordLength);
        klaunch_printf("vectorWidth: %d\n", this->VectorWidth);
        additionalDefines += 
                DefineGen.makePasswordNoMemMultipleIncrementorsLE(
                this->passwordLength, this->VectorWidth, 
                this->Charset->getCharset(), 1, 4);
    }
    
    return additionalDefines;
}