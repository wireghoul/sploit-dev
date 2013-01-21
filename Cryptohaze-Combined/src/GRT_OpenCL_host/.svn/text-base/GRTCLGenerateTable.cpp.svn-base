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

#include "GRT_OpenCL_host/GRTCLGenerateTable.h"
#include "GRT_Common/GRTCommon.h"
#include "GRT_Common/GRTTableHeaderV1.h"
#include "GRT_Common/GRTTableSearchV1.h"
#include "GRT_Common/GRTTableHeaderV2.h"
#include "GRT_Common/GRTTableSearchV2.h"
#include "GRT_Common/GRTTableHeaderV3.h"
#include "GRT_Common/GRTTableSearchV3.h"
#include "GRT_Common/GRTTableHeaderVWeb.h"
#include "GRT_Common/GRTHashes.h"
#include "GRT_OpenCL_host/GRTCLUtils.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <curl/curl.h>

#include "CH_Common/Timer.h"
#include "OpenCL_Common/GRTOpenCL.h"

#ifdef _WIN32
#include <direct.h>
#endif

using namespace std;

static void bfi_int_magic(unsigned char *kernel, int len);


// Handle a read pf the table header
size_t table_upload_write(void *buffer, size_t size, size_t nmemb, void *userp) {

    // Get us a vector pointer.
    std::vector<uint8_t> *returnBuffer = (std::vector<uint8_t> *)userp;
    uint8_t *bufferPointer = (uint8_t *)buffer;

    // Allocate as much space as we need
    returnBuffer->reserve(returnBuffer->size() + (size * nmemb));

    for (int i = 0; i < (size * nmemb); i++) {
        returnBuffer->push_back(bufferPointer[i]);
    }
    //printf("Size received: %d\n", size * nmemb);

    return (size * nmemb);
}

// Generates the initial passwords.  This works for my hash functions...
void GRTCLGenerateTable::generateInitialPasswords(int passwordType) {
    if (!this->TableParameters || !this->Charset) {
        printf("Must call setGRTGenCommandLineData and setGRTCharsetSingle before calling generateInitialPasswords!\n");
        exit(1);
    }
    if (!this->HOST_End_Hashes || !this->HOST_Initial_Passwords) {
        printf("Must call mallocDeviceAndHostSpace first!\n");
        exit(1);
    }
    if (!this->CurrentTableHeader) {
        printf("Error: Cannot call generateInitialPasswords without setting table header!\n");
        exit(1);
    }
    if (passwordType != PASSWORD_TYPE_NORMAL) {
        printf("Only normal passwords supported!\n");
        exit(1);
    }

    uint32_t chain_index, password_position, character_offset;
    char character;
    char *Charset;
    int CharsetLength, i;

    // To get the charset from the table header.
    char *charsetLengths;
    char **charsetFullArray;

    charsetLengths = this->CurrentTableHeader->getCharsetLengths();
    charsetFullArray = this->CurrentTableHeader->getCharset();

    //Charset = this->Charset->getCharset();
    //CharsetLength = this->Charset->getCharsetLength();
    // Update to use charset from table header
    Charset = charsetFullArray[0];
    CharsetLength = charsetLengths[0];

    printf("Charset length: %d\n", CharsetLength);
    // This becomes slightly complex.  In order to work properly with the GPU
    // memory coalescing, we interleave by 32-bit word.

    // Zero the password space.  This is 16 bytes, for MD5/MD4/SHA1/etc.
    memset(this->HOST_Initial_Passwords, 0, this->PasswordInputBlockLengthBytes * this->CurrentTableHeader->getNumberChains() * sizeof(unsigned char));

    for (chain_index = 0; chain_index < this->CurrentTableHeader->getNumberChains(); chain_index++) {
        // Run through the password length.
        for (password_position = 0; password_position < this->CurrentTableHeader->getPasswordLength(); password_position++) {
            //character = Charset[mt_lrand() % CharsetLength];
            character = Charset[this->RandomGenerator->getRandomValue() % CharsetLength];
            // Calculate the interleave offset.  Note we are working in 32-bit words (4 characters)
            character_offset = (chain_index * 4) // Initial offset based on the chain index.
                    + (password_position % 4) // The position within the password "chunk" of 4
                    + (password_position / 4) * (4 * this->CurrentTableHeader->getNumberChains());  // The offset of the chunk
            this->HOST_Initial_Passwords[character_offset] = character;
            //printf("Chain: %d  PassPos: %d  Char: %c  Offset: %d\n", chain_index, password_position, character, character_offset);
        }
    }
    
    // Clean up the charset we got from the table header.
    delete[] charsetLengths;

    for (i = 0; i < 16; i++)
        delete[] charsetFullArray[i];
    delete[] charsetFullArray;
}

void GRTCLGenerateTable::mallocDeviceAndHostSpace() {
    uint64_t deviceMemoryBytes;
    uint32_t bytesPerChain;
    uint32_t numberChainsToTry;

    cl_int errorCode;
    
    if ((this->TableParameters == NULL) || (this->Charset == NULL)) {
        printf("Must call setGRTGenCommandLineData and setGRTCharsetSingle before calling generateInitialPasswords!\n");
        exit(1);
    }
    if (!this->CurrentTableHeader) {
        printf("Error: Cannot call generateInitialPasswords without setting table header!\n");
        exit(1);
    }

    // If the number of chains is not set, auto-generate it.
    if (this->CurrentTableHeader->getNumberChains() == 0) {
        deviceMemoryBytes = this->OpenCL->getMaximumAllocSize();

        printf("Attempting to auto-set numberChains for GPU RAM\n");
        printf("GPU has %lu bytes of RAM\n", deviceMemoryBytes);

        // Determine the number of bytes per chain total
        // Later, we can chop this to try multiple 128MB or greater mallocs.
        bytesPerChain = (this->PasswordInputBlockLengthBytes +
                this->HashOutputBlockLengthBytes) * sizeof(unsigned char);

        // Ballpark the number of chains to try.
        // This is limited by malloc size, so let's get creative.
        numberChainsToTry = (uint32_t)(1.0 * (float)(deviceMemoryBytes / bytesPerChain));


        for (numberChainsToTry; numberChainsToTry > 0; numberChainsToTry -= 1000000) {
            // Try to allocate the memory - if it fails, reduce the number of chains.
            this->DEVICE_Initial_Passwords = clCreateBuffer (this->OpenCL->getContext(),
                    CL_MEM_READ_WRITE,
                    this->PasswordInputBlockLengthBytes * numberChainsToTry * sizeof(unsigned char),
                    NULL,
                    &errorCode);

            if (errorCode != CL_SUCCESS) {
                // Alloc failed.  Go around for another attempt.
                continue;
            } else {
                // Alloc succeeded.
                this->DEVICE_End_Hashes = clCreateBuffer (this->OpenCL->getContext(),
                    CL_MEM_WRITE_ONLY,
                    this->HashOutputBlockLengthBytes * numberChainsToTry * sizeof(unsigned char),
                    NULL,
                    &errorCode);
                if (errorCode != CL_SUCCESS) {
                    // Alloc failed.  Remove the initial passwords and go around.
                    clReleaseMemObject(this->DEVICE_Initial_Passwords);
                    continue;
                } else {
                    // The second alloc succeeded: everything is good!

                    // As the number of chains MUST be a multiple of vector_width,
                    // make this adjustment & set it.

                    // If it is not even, reset it.
                    if (numberChainsToTry % this->TableParameters->getVectorWidth()) {
                        numberChainsToTry -= (numberChainsToTry % this->TableParameters->getVectorWidth());
                    }

                    this->TableParameters->setNumberChains(numberChainsToTry);
                    // Exit the for loop.
                    break;
                }

            }
        }

    printf("Using %lu chains in GPU RAM\n", this->TableParameters->getNumberChains());
    } else {
        // Eheheh... if I'm *not* auto generating length, I should probably alloc
        // the data in that path too.  Whoops!

        numberChainsToTry = this->CurrentTableHeader->getNumberChains();

        if (numberChainsToTry % this->TableParameters->getVectorWidth()) {
            printf("WARNING: numberChains not a multiple of vector_width.  Resetting...\n");
            numberChainsToTry -= (numberChainsToTry % this->TableParameters->getVectorWidth());
            printf("Reset numberChains to %u\n", numberChainsToTry);
            this->CurrentTableHeader->setNumberChains(numberChainsToTry);
        }

        this->DEVICE_Initial_Passwords = clCreateBuffer (this->OpenCL->getContext(),
            CL_MEM_READ_WRITE,
            this->PasswordInputBlockLengthBytes * this->CurrentTableHeader->getNumberChains() * sizeof(unsigned char),
            NULL,
            &errorCode);
        
        if (errorCode != CL_SUCCESS) {
            printf("ERROR: Cannot allocate GPU memory.  Try a smaller number of chains.\n");
            printf("Error: %s\n", print_cl_errstring(errorCode));
            exit(1);
        }
        this->DEVICE_End_Hashes = clCreateBuffer (this->OpenCL->getContext(),
            CL_MEM_WRITE_ONLY,
            this->HashOutputBlockLengthBytes * this->CurrentTableHeader->getNumberChains() * sizeof(unsigned char),
            NULL,
            &errorCode);

        if (errorCode != CL_SUCCESS) {
            printf("ERROR: Cannot allocate GPU memory.  Try a smaller number of chains.\n");
            printf("Error: %s\n", print_cl_errstring(errorCode));
            exit(1);
        }
    }
    // Either way, we need the charset allocated.
    this->DEVICE_Charset = clCreateBuffer (this->OpenCL->getContext(),
        CL_MEM_READ_ONLY,
        512 * sizeof(unsigned char),
        NULL,
        &errorCode);

    if (errorCode != CL_SUCCESS) {
        printf("ERROR: Cannot allocate GPU memory.  Try a smaller number of chains.\n");
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    // If we're here, that means both GPU allocs succeeded.  Host alloc time!
    this->HOST_Initial_Passwords = (unsigned char*) malloc(this->PasswordInputBlockLengthBytes * this->CurrentTableHeader->getNumberChains() * sizeof(unsigned char));
    this->HOST_End_Hashes = (unsigned char*) malloc(this->HashOutputBlockLengthBytes * this->CurrentTableHeader->getNumberChains() * sizeof(unsigned char));

    if (!this->HOST_End_Hashes || !this->HOST_Initial_Passwords) {
        printf("ERROR: Cannot allocate host memory.  Try a smaller number of chains.\n");
        exit(1);
    }
}

void GRTCLGenerateTable::freeDeviceAndHostSpace() {
    // I think this should do it...
    clReleaseMemObject(this->DEVICE_Initial_Passwords);
    clReleaseMemObject(this->DEVICE_End_Hashes);
    clReleaseMemObject(this->DEVICE_Charset);
    free(this->HOST_Initial_Passwords);
    free(this->HOST_End_Hashes);
}

GRTCLGenerateTable::GRTCLGenerateTable(int hashLengthBytes, int passwordLengthBytes) {
    int roundTemp;

    // Assume, for now, this is a multiple of 4 as it should be.
    this->HashOutputBlockLengthBytes = hashLengthBytes;

    // Round the password length up to the nearest multiple of 4.
    roundTemp = passwordLengthBytes / 4;
    // If it is not a multiple, add one and remultiply.
    if (passwordLengthBytes % 4) {
        roundTemp++;
    }
    this->PasswordInputBlockLengthBytes = roundTemp * 4;
    // Null these out until they are set.
    this->TableParameters = NULL;
    this->Charset = NULL;

    this->HOST_Initial_Passwords = NULL;
    this->HOST_End_Hashes = NULL;

    this->CurrentTableHeader = NULL;

    printf("GRTCLGenerateTable: Hash: %d  Pass: %d\n", this->HashOutputBlockLengthBytes, this->PasswordInputBlockLengthBytes);
}

GRTCLGenerateTable::~GRTCLGenerateTable() {

}

void GRTCLGenerateTable::setGRTCLGenCommandLineData(GRTCLGenCommandLineData *NewTableParameters) {
    this->TableParameters = NewTableParameters;
}

void GRTCLGenerateTable::setGRTCLCharsetSingle(GRTCharsetSingle *NewCharset) {
    this->Charset = NewCharset;
}


void GRTCLGenerateTable::createTables() {
    char *charset;
    char *charsetLengths;
    char **charsetFullArray;
    char HOST_Charset[512];
    UINT4 HOST_Charset_Length = 0;
    int i, j;

    uint64_t ChainsCompleted = 0;
    // Where we are in the current chain
    uint64_t CurrentChainStartOffset = 0;
    // Calculated on the host for modulus reasons
    uint64_t CharsetOffset = 0;
    uint64_t PasswordSpaceOffset = 0;
    uint64_t StepsPerInvocation = 100;
    uint64_t ActiveThreads = 0;

    UINT4 tableNumber = 0;

    char filenameBuffer[4096];

    uint64_t Number_Of_Threads;

    uint32_t NumberOfChainsToRun;
    // For calculating speeds - default kernels are 1 wide.
    int vectorWidth = 1;


    // Table header & table files for the output
    GRTTableHeader *TableHeader = NULL;
    GRTTableSearch *TableOutput = NULL;

    hashPasswordData chainToWrite;

    GRTHashes HashNames;

    cl_program GenerateTableProgram;
    cl_kernel GenerateTableKernel;
    cl_command_queue GenerateTableCommandQueue;
    cl_int errorCode = CL_SUCCESS;
    
    cl_uint deviceChainLength;
    cl_uint deviceNumberOfChains;
    cl_uint deviceTableIndex;
    size_t numberWorkgroups;
    size_t numberWorkitems;

    cl_event kernelLaunchEvent;

    Timer kernelLaunchTimer;
    Timer fullRunTimer;

    char buildOptions[1024];

    // For V3 tables: How many chains we've generated so far.
    uint64_t chainStartOffset = 0;

    // Get the command line workgroup size data.
    numberWorkgroups = this->TableParameters->getWorkgroups();
    numberWorkitems = this->TableParameters->getWorkitems();

    // If it has not been set, use sane defaults.
    if (numberWorkgroups == 0) {
        numberWorkgroups = this->OpenCL->getDefaultBlockCount() * this->OpenCL->getDefaultThreadCount();
    }
    if (numberWorkitems == 0) {
        numberWorkitems = this->OpenCL->getDefaultThreadCount();
    }
    
    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }

    this->OpenCL->createCommandQueue();
    GenerateTableCommandQueue = this->OpenCL->getCommandQueue();
    



    // Set up the timer we use for kernel times.
    //cutCreateTimer(&timer);

    // Make the parts directory.
#ifdef _WIN32
    _mkdir("./parts");
#else
    mkdir("./parts", S_IRWXO | S_IXOTH | S_IRWXU | S_IRWXG);
#endif

        // Ready to start making tables.
    for (tableNumber = 0; tableNumber < this->TableParameters->getNumberTables(); tableNumber++) {
        // Yes, we 1-index here.
        printf("\n\nCreating table %d of %d\n\n", tableNumber + 1, this->TableParameters->getNumberTables());
        
        if (this->TableParameters->getUseWebGenerate()) {
            // Set up web gen headers
            GRTTableHeader *WebTable = new GRTTableHeaderVWeb();

            WebTable->setWebURL(this->TableParameters->getWebGenURL());
            WebTable->setWebUsername(this->TableParameters->getWebUsername());
            WebTable->setWebPassword(this->TableParameters->getWebPassword());

            WebTable->readTableHeader("generate");
            printf("Web Generate Table header:\n");
            WebTable->printTableHeader();

            TableOutput = new GRTTableSearchV3();
            TableHeader = new GRTTableHeaderV3();
            printf("Copying tables...\n");
            TableHeader->setHeaderString(WebTable->getHeaderString());
            //TableHeader->printTableHeader();

            sprintf(filenameBuffer, "parts/WebGen-%s-len%d-idx%d-chr%d-cl%d-sd%lu-%d-v%d.part",
                HashNames.GetHashStringFromId(TableHeader->getHashVersion()), TableHeader->getPasswordLength(), TableHeader->getTableIndex(),
                0, TableHeader->getChainLength(),
                (uint64_t)TableHeader->getRandomSeedValue(), tableNumber, TableHeader->getTableVersion());
            delete WebTable;
        } else {
            switch(this->TableParameters->getOutputTableVersion()) {
                case 1:
                    printf("Creating GRTV1 output table.\n");
                    TableOutput = new GRTTableSearchV1();
                    TableHeader = new GRTTableHeaderV1();
                    TableHeader->setTableVersion(0x01);
                    break;
                case 2:
                    printf("Creating GRTV2 output table.\n");
                    TableOutput = new GRTTableSearchV2();
                    TableHeader = new GRTTableHeaderV2();
                    TableHeader->setTableVersion(0x02);
                    break;
                case 3:
                    printf("Creating GRTV3 output table.\n");
                    TableOutput = new GRTTableSearchV3();
                    TableHeader = new GRTTableHeaderV3();
                    TableHeader->setTableVersion(0x03);
                    break;
                default:
                    // This should NEVER happen... it's checked in the initial checking.
                    printf("Unknown output table version %d!\n", this->TableParameters->getOutputTableVersion());
                    exit(1);
            }
        
            TableHeader->setHashVersion(this->TableParameters->getHashType());
            TableHeader->setHashName((char *)HashNames.GetHashStringFromId(this->TableParameters->getHashType()));

            TableHeader->setChainLength(this->TableParameters->getChainLength());
            TableHeader->setIsPerfect(0);
            TableHeader->setTableIndex(this->TableParameters->getTableIndex());
            TableHeader->setNumberChains(this->TableParameters->getNumberChains());
            TableHeader->setPasswordLength(this->TableParameters->getPasswordLength());

            // Set the random seed for V3 tables
            TableHeader->setRandomSeedValue(this->TableParameters->getRandomSeed());
            TableHeader->setChainStartOffset(chainStartOffset);

            // Set up the charset
            TableHeader->setCharsetCount(1);
            char CharsetLengths[16];
            memset(CharsetLengths, 0, sizeof(CharsetLengths));
            CharsetLengths[0] = this->Charset->getCharsetLength();
            TableHeader->setCharsetLengths(CharsetLengths);

            char **CharsetArray = new char*[16];
            CharsetArray[0] = this->Charset->getCharset();

            TableHeader->setCharset(CharsetArray);

            // If this is a V2 table, set bits in hash.
            if (this->TableParameters->getOutputTableVersion() == 2) {
                int bitsInHash;
                int bitsInPassword;
                int bytesInHashOutputBlock;

                bytesInHashOutputBlock = this->HashOutputBlockLengthBytes;

                // We trim to 16 bytes - keeping a full SHA1 around is silly.
                if (bytesInHashOutputBlock > 16) {
                    bytesInHashOutputBlock = 16;
                }

                bitsInPassword = TableHeader->getBitsInPassword();

                //printf("Bits in password: %d\n", bitsInPassword);

                // Set to total length of both.
                bitsInHash = (bytesInHashOutputBlock * 8) + TableHeader->getBitsInPassword();

                //printf("bitsInHash: %d\n", bitsInHash);

                // Round down to the nearest byte.

                bitsInHash /= 8;
                bitsInHash *= 8;

                bitsInHash -= TableHeader->getBitsInPassword();

                printf("Generating V2 table with %d bits in hash.\n", bitsInHash);
                TableHeader->setBitsInHash(bitsInHash);
            }

            // If this is a V3 table, set bits in hash.
            // Hardcoded to 64 right now
            // todo: Take command line input on this!
            if (this->TableParameters->getOutputTableVersion() == 3) {
                TableHeader->setBitsInHash(this->TableParameters->getOutputTableBits());
            }

            // Build a filename.
            printf("Random seed: %lu\n", this->TableParameters->getRandomSeed());
            sprintf(filenameBuffer, "parts/%s-len%d-idx%d-chr%d-cl%d-sd%lu-%d-v%d.part",
                HashNames.GetHashStringFromId(this->TableParameters->getHashType()), this->TableParameters->getPasswordLength(), this->TableParameters->getTableIndex(),
                this->Charset->getCharsetLength(), this->TableParameters->getChainLength(),
                (uint64_t)this->TableParameters->getRandomSeed(), tableNumber, this->TableParameters->getOutputTableVersion());

            printf("Output to: %s\n", filenameBuffer);
        }
        // At this point, the table header is generated.  Malloc host & device space.
        // This is done here to deal with different table sizes/etc from WebGen.
        // Previously, it was done at the beginning, as all tables would be the same size.
        // Efficiency loss slight, sanity boost, massive.
        this->CurrentTableHeader = TableHeader;
        TableHeader->printTableHeader();

        sprintf(buildOptions, "-D PASSWORD_LENGTH=%d -I .  -I ./kernels", this->CurrentTableHeader->getPasswordLength());
    
        // Try for the kernel
        if (this->TableParameters->getUseAmdKernels()) {
            // Use BFI_INT - pass in define for it, and apply patch.
            vectorWidth = this->TableParameters->getVectorWidth();
            sprintf(buildOptions, "%s -D VECTOR_WIDTH=%d", buildOptions, vectorWidth);
            sprintf(buildOptions, "%s -D BITALIGN", buildOptions);
            this->OpenCL->buildProgramFromManySourcesConcat(this->getHashFileName(), buildOptions, this->getKernelSourceString());
            this->OpenCL->doAMDBFIPatch();
        } else {
            // No BFI_INT support - don't do anything fancy.
            vectorWidth = this->TableParameters->getVectorWidth();
            sprintf(buildOptions, "%s -D VECTOR_WIDTH=%d", buildOptions, vectorWidth);
            this->OpenCL->buildProgramFromManySourcesConcat(this->getHashFileName(), buildOptions, this->getKernelSourceString());
        }

        GenerateTableProgram = this->OpenCL->getProgram();
        GenerateTableKernel = clCreateKernel (GenerateTableProgram, this->getHashKernelName().c_str(), &errorCode);

        
        this->mallocDeviceAndHostSpace();

        // Set up the charset for the GPU
        charsetLengths = TableHeader->getCharsetLengths();
        charsetFullArray = TableHeader->getCharset();

        //charset = this->Charset->getCharset();
        //HOST_Charset_Length = this->Charset->getCharsetLength();
        charset = charsetFullArray[0];
        HOST_Charset_Length = charsetLengths[0];

        for (i = 0; i < 512; i++) {
            HOST_Charset[i] = charset[i % HOST_Charset_Length];
            //printf("%d: %c\n", i, HOST_Charset[i]);
        }
        // Clean up the charset we got from the table header.
        delete[] charsetLengths;

        for (i = 0; i < 16; i++)
            delete[] charsetFullArray[i];
        delete[] charsetFullArray;

        Number_Of_Threads = numberWorkgroups;
        printf("Number of threads: %d\n", Number_Of_Threads);
        // Copy all the constants to the GPU memory
        // Copy the charset to the device
        errorCode = clEnqueueWriteBuffer (GenerateTableCommandQueue,
                this->DEVICE_Charset,
                CL_TRUE /* blocking write */,
                0 /* offset */,
                512 /* bytes to copy */,
                (void *)HOST_Charset,
                NULL, NULL, NULL /* event list stuff */);
        if (errorCode != CL_SUCCESS) {
            printf("Error: %s\n", print_cl_errstring(errorCode));
            exit(1);
        }


        deviceChainLength = this->CurrentTableHeader->getChainLength();
        deviceNumberOfChains = this->CurrentTableHeader->getNumberChains();
        deviceTableIndex = this->CurrentTableHeader->getTableIndex();

        errorCode = 0;
        /* __constant unsigned char *deviceCharset */
        errorCode |= clSetKernelArg (GenerateTableKernel, 0, sizeof(cl_mem), &this->DEVICE_Charset);
        /* __constant unsigned int deviceCharsetLength */
        errorCode |= clSetKernelArg (GenerateTableKernel, 1, sizeof(cl_uint), &HOST_Charset_Length);
        /* __constant unsigned int deviceChainLength */
        errorCode |= clSetKernelArg (GenerateTableKernel, 2, sizeof(cl_uint), &deviceChainLength);
        /* __constant unsigned int deviceNumberOfChains */
        errorCode |= clSetKernelArg (GenerateTableKernel, 3, sizeof(cl_uint), &deviceNumberOfChains);
        /* __constant unsigned int deviceTableIndex */
        errorCode |= clSetKernelArg (GenerateTableKernel, 4, sizeof(cl_uint), &deviceTableIndex);
        /* __constant unsigned int deviceNumberOfThreads */
        errorCode |= clSetKernelArg (GenerateTableKernel, 5, sizeof(cl_uint), &Number_Of_Threads);
        /* __global   unsigned int *initialPasswordArray */
        errorCode |= clSetKernelArg (GenerateTableKernel, 6, sizeof(cl_mem), &this->DEVICE_Initial_Passwords);
        /* __global   unsigned int *outputHashArray */
        errorCode |= clSetKernelArg (GenerateTableKernel, 7, sizeof(cl_mem), &this->DEVICE_End_Hashes);
        // The rest are iteration specific, so will have to wait.

        if (errorCode != CL_SUCCESS) {
            printf("Error: %s\n", print_cl_errstring(errorCode));
            exit(1);
        }

        
        // Set the random sequence for WebGen tables
        if (this->TableParameters->getUseWebGenerate()) {
            this->RandomGenerator->setSeed(TableHeader->getRandomSeedValue());
            this->RandomGenerator->skipSome(TableHeader->getChainStartOffset() * TableHeader->getPasswordLength());
        }
        
        // Do the table output through the table class.
        TableOutput->setTableHeader(TableHeader);
        TableOutput->openOutputFile(filenameBuffer);


        this->generateInitialPasswords(0);
        errorCode = clEnqueueWriteBuffer (GenerateTableCommandQueue,
            this->DEVICE_Initial_Passwords,
            CL_TRUE /* blocking write */,
            0 /* offset */,
            this->PasswordInputBlockLengthBytes * this->CurrentTableHeader->getNumberChains() * sizeof(unsigned char) /* bytes to copy */,
            this->HOST_Initial_Passwords,
            NULL, NULL, NULL /* event list stuff */);
        if (errorCode != CL_SUCCESS) {
            printf("Error: %s\n", print_cl_errstring(errorCode));
            exit(1);
        }

        // Pre-kernel init.
        ChainsCompleted = 0;
        CharsetOffset = 0;
        PasswordSpaceOffset = 0;
        StepsPerInvocation = 100;

        // If kernel time is set to zero, run full chains.
        if (!this->TableParameters->getKernelTimeMs()) {
            StepsPerInvocation = this->CurrentTableHeader->getChainLength();
        }

        NumberOfChainsToRun = this->CurrentTableHeader->getNumberChains();

        NumberOfChainsToRun /= vectorWidth;

        fullRunTimer.start();

        // While we haven't finished all the chains:
        while (ChainsCompleted < NumberOfChainsToRun) {
            CurrentChainStartOffset = 0;

            while (CurrentChainStartOffset < this->CurrentTableHeader->getChainLength()) {
                // Calculate the right charset offset
                CharsetOffset = CurrentChainStartOffset % HOST_Charset_Length;
                PasswordSpaceOffset = (ChainsCompleted / (Number_Of_Threads));

                // Don't overrun the end of the chain
                if ((CurrentChainStartOffset + StepsPerInvocation) > this->CurrentTableHeader->getChainLength()) {
                    StepsPerInvocation = this->CurrentTableHeader->getChainLength() - CurrentChainStartOffset;
                }

                // Copy the remaining data to the kernel params
                errorCode = 0;
                /* __global   unsigned int passwordSpaceOffset */
                errorCode |= clSetKernelArg (GenerateTableKernel, 8, sizeof(cl_uint), &PasswordSpaceOffset);
                /* __global   unsigned int startChainIndex */
                errorCode |= clSetKernelArg (GenerateTableKernel, 9, sizeof(cl_uint), &CurrentChainStartOffset);
                /* __global   unsigned int stepsToRun */
                errorCode |= clSetKernelArg (GenerateTableKernel, 10, sizeof(cl_uint), &StepsPerInvocation);
                /* __global   unsigned int charsetOffset */
                errorCode |= clSetKernelArg (GenerateTableKernel, 11, sizeof(cl_uint), &CharsetOffset);

                if (errorCode != CL_SUCCESS) {
                    printf("Error: %s\n", print_cl_errstring(errorCode));
                    exit(1);
                }

                kernelLaunchTimer.start();
                // Actually launch the kernel!
                errorCode = clEnqueueNDRangeKernel(GenerateTableCommandQueue,
                        GenerateTableKernel,
                        1 /* numDims */,
                        NULL /* offset */,
                        &numberWorkgroups,
                        &numberWorkitems,
                        NULL, NULL, 
                        &kernelLaunchEvent);

                if (errorCode != CL_SUCCESS) {
                    printf("\nError on wait for event!\n");
                    fflush(stdout);
                };

                if (clWaitForEvents(1, &kernelLaunchEvent) != CL_SUCCESS) {
                    printf("\nError on wait for event!\n");
                    fflush(stdout);
                };
                
                // Release the event to prevent memory leaks.
                clReleaseEvent(kernelLaunchEvent);

                double ref_time = kernelLaunchTimer.stop() * (double)1000.0;

				// Deal with timer tick resolution: If the timer is less than 10ms, set to 10ms.
				// This deals with a 15.6ms timer res on Windows...
				if (ref_time < 10) {
					ref_time = 10.0;
				}
				
                // For actual rate generation, we need to know how many threads are doing real work.
                ActiveThreads = (Number_Of_Threads);
                if ((NumberOfChainsToRun - ChainsCompleted) < ActiveThreads) {
                    ActiveThreads = (/*this->CurrentTableHeader->getNumberChains()*/NumberOfChainsToRun - ChainsCompleted);
                    
                }

                printf("Kernel Time: %0.03f ms  Step rate: %0.2f M/s Done: %0.2f%%  \r",
                    ref_time,

                    (((float)vectorWidth * (float)ActiveThreads * (float)StepsPerInvocation) / (float)1000.0) / (float)ref_time,
                    
                    (float)(100 * ((float)((float)ChainsCompleted * (float)this->CurrentTableHeader->getChainLength() +
                        (float)CurrentChainStartOffset * (float)ActiveThreads) /
                        (float)((float)NumberOfChainsToRun * (float)this->CurrentTableHeader->getChainLength()))));
                fflush(stdout);
                CurrentChainStartOffset += StepsPerInvocation;
                if (this->TableParameters->getKernelTimeMs()) {
                    // Adjust the steps per invocation if needed.
                    if ((ref_time > 1.1 * this->TableParameters->getKernelTimeMs()) || (ref_time < 0.9 * this->TableParameters->getKernelTimeMs())) {
                        StepsPerInvocation = (UINT4)((float)StepsPerInvocation * ((float)this->TableParameters->getKernelTimeMs() / ref_time));
						if (StepsPerInvocation < 100) {
							StepsPerInvocation = 100;
						}
						//printf("\nKernel time goal: %d\n",this->CurrentTableHeader->getKernelTimeMs()); 
						//printf("\nStepsPerInvocation: %d\n", StepsPerInvocation);
                    }
                }
                
            }
            ChainsCompleted += (Number_Of_Threads);
        }
        printf("\n\nDone chain running\n");
        printf("Total execution rate: %0.2f M/s\n", (float)((uint64_t)this->CurrentTableHeader->getNumberChains() *
                (uint64_t)this->CurrentTableHeader->getChainLength()) / (fullRunTimer.stop() * 1000000));

        errorCode = clEnqueueReadBuffer (GenerateTableCommandQueue,
            this->DEVICE_End_Hashes,
            CL_TRUE /* blocking write */,
            0 /* offset */,
            this->HashOutputBlockLengthBytes * this->CurrentTableHeader->getNumberChains() * sizeof(unsigned char) /* bytes to copy */,
            this->HOST_End_Hashes,
            NULL, NULL, NULL /* event list stuff */);
        if (errorCode != CL_SUCCESS) {
            printf("Error: %s\n", print_cl_errstring(errorCode));
            exit(1);
        }


        printf("Readback from device complete\n\n");
        
        if (0) {
            for (i = 0; i < this->CurrentTableHeader->getNumberChains(); i++) {
                UINT4 base_offset;

                // Print the end hashes
                printf("Chain %d: ", i);
                // Print the password
                base_offset = this->CurrentTableHeader->getNumberChains() * 0; // Segment start
                base_offset += i * 4; // This chain start
                printf("%c%c%c%c", HOST_Initial_Passwords[base_offset], HOST_Initial_Passwords[base_offset + 1], HOST_Initial_Passwords[base_offset + 2], HOST_Initial_Passwords[base_offset + 3]);
                base_offset = this->CurrentTableHeader->getNumberChains() * 1 * 4; // Segment start
                base_offset += i * 4; // This chain start
                printf("%c%c%c%c  ", HOST_Initial_Passwords[base_offset], HOST_Initial_Passwords[base_offset + 1], HOST_Initial_Passwords[base_offset + 2], HOST_Initial_Passwords[base_offset + 3]);

                base_offset = this->CurrentTableHeader->getNumberChains() * 0; // Segment start
                base_offset += i * 4; // This chain start
                printf("%02X%02X%02X%02X", HOST_End_Hashes[base_offset], HOST_End_Hashes[base_offset + 1], HOST_End_Hashes[base_offset + 2], HOST_End_Hashes[base_offset + 3]);
                base_offset = this->CurrentTableHeader->getNumberChains() * 1 * 4; // Segment start
                base_offset += i * 4; // This chain start
                printf("%02X%02X%02X%02X", HOST_End_Hashes[base_offset], HOST_End_Hashes[base_offset + 1], HOST_End_Hashes[base_offset + 2], HOST_End_Hashes[base_offset + 3]);
                base_offset = this->CurrentTableHeader->getNumberChains() * 2 * 4; // Segment start
                base_offset += i * 4; // This chain start
                printf("%02X%02X%02X%02X", HOST_End_Hashes[base_offset], HOST_End_Hashes[base_offset + 1], HOST_End_Hashes[base_offset + 2], HOST_End_Hashes[base_offset + 3]);
                base_offset = this->CurrentTableHeader->getNumberChains() * 3 * 4; // Segment start
                base_offset += i * 4; // This chain start
                printf("%02X%02X%02X%02X", HOST_End_Hashes[base_offset], HOST_End_Hashes[base_offset + 1], HOST_End_Hashes[base_offset + 2], HOST_End_Hashes[base_offset + 3]);
                printf("\n");
            }
        }

        // Create a vector of data... trying this C++ methods...

        vector<hashPasswordData> tableData;
        hashPasswordData tableElement;

        for (i = 0; i < this->CurrentTableHeader->getNumberChains(); i++) {
            uint32_t array_offset;
            // Clear the structure completely
            memset(&tableElement, 0, sizeof(tableElement));
            for (j = 0; j < this->HashOutputBlockLengthBytes; j++) {
                array_offset = (i * 4) // Initial offset based on the chain index.
                  + ((j) % 4) // The position within the hash "chunk" of 4
                  + (j / 4) * (4 * this->CurrentTableHeader->getNumberChains());  // The offset of the chunk
                tableElement.hash[j] = this->HOST_End_Hashes[array_offset];
            }
            
            for (j = 0; j < 16; j++) {
                array_offset = (i * 4) // Initial offset based on the chain index.
                  + ((j) % 4) // The position within the hash "chunk" of 4
                  + (j / 4) * (4 * this->CurrentTableHeader->getNumberChains());  // The offset of the chunk
                tableElement.password[j] = this->HOST_Initial_Passwords[array_offset];
            }
            // Add to the vector
            tableData.push_back(tableElement);
        }

        // Sort the vector.  We do NOT SORT v3 tables!
        if (this->CurrentTableHeader->getTableVersion() != 3) {
            std::sort(tableData.begin(), tableData.end(), tableDataSortPredicate);
        }



        for (uint64_t password = 0; password < this->CurrentTableHeader->getNumberChains(); password++) {
            memset(&chainToWrite, 0, sizeof(hashPasswordData));
            memcpy(chainToWrite.hash, &tableData[password].hash, 16);
            memcpy(chainToWrite.password, &tableData[password].password, 16);
            TableOutput->writeChain(&chainToWrite);
        }

        TableOutput->closeOutputFile();

        if (this->TableParameters->getUseWebGenerate()) {
            // Upload the table.  If it fails, bail out.
            if (!this->uploadWebGenTableFile(filenameBuffer)) {
                delete TableOutput;
                delete TableHeader;
                this->CurrentTableHeader = NULL;
                this->freeDeviceAndHostSpace();
                return;
            }
            unlink(filenameBuffer);
        }

        // Add how many chains we've generated to the start offset
        chainStartOffset += this->CurrentTableHeader->getNumberChains();

        delete TableOutput;
        delete TableHeader;
        this->CurrentTableHeader = NULL;
        this->freeDeviceAndHostSpace();
    }
}

void GRTCLGenerateTable::setOpenCL(CryptohazeOpenCL* newOpenCL) {
    this->OpenCL = newOpenCL;
}

static int progress(void *p,
                    double dltotal, double dlnow,
                    double ultotal, double ulnow) {
    printf("Uploaded: %0.2f%%\r", 100.0 * ulnow / ultotal);
    fflush(stdout);
    return 0;
}

int GRTCLGenerateTable::uploadWebGenTableFile(char *filename) {
    printf("Uploading table file %s\n", filename);

    CURL *curl;
    CURLcode res;
    // structures to build the post
    struct curl_httppost* post = NULL;
    struct curl_httppost* last = NULL;
    std::vector <uint8_t> returnBuffer;

    curl = curl_easy_init();
    if (curl) {

        // Add the elements to the form.
        curl_formadd(&post, &last, CURLFORM_COPYNAME, "uploadFilename",
            CURLFORM_FILE, filename, CURLFORM_END);
        curl_formadd(&post, &last, CURLFORM_COPYNAME, "uploadFilenameString",
            CURLFORM_PTRCONTENTS, filename, CURLFORM_END);


        curl_easy_setopt(curl, CURLOPT_URL, this->TableParameters->getWebGenURL().c_str());
        // Add the post data
        curl_easy_setopt(curl, CURLOPT_HTTPPOST, post);
        // Pass a pointer to our tableValid variable for the callback.
        //curl_easy_setopt(curl, CURLOPT_WRITEDATA, &returnBuffer);
        //curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, table_search_write);
        curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, progress);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        // If we have a username, set username/password/authentication.
        if (this->TableParameters->getWebUsername().length()) {
            curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
            curl_easy_setopt(curl, CURLOPT_USERNAME, this->TableParameters->getWebUsername().c_str());
            curl_easy_setopt(curl, CURLOPT_PASSWORD, this->TableParameters->getWebPassword().c_str());
        }
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &returnBuffer);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, table_upload_write);

        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            // Error: Something is wrong.
            printf("curl error: %s\n", curl_easy_strerror(res));
            curl_easy_cleanup(curl);
            curl_formfree(post);
            return 0;
        }

        printf("\n\n");
        printf("Upload status: ");
        for (int charpos = 0; charpos < returnBuffer.size(); charpos++) {
            printf("%c", returnBuffer[charpos]);
        }

        // If we have failed, terminate the task.
        if ((returnBuffer.size() == 0) || (returnBuffer.at(0) != 'O') || (returnBuffer.at(1) != 'K')) {
            printf("Table generation failure!  Exiting!\n");
            unlink(filename);
            curl_easy_cleanup(curl);
            curl_formfree(post);
            return 0;
        } else {
            printf("\nTable uploaded successfully.\n");
        }

        /* always cleanup */
        curl_easy_cleanup(curl);
        curl_formfree(post);
        post = NULL;
        last = NULL;
        return 1;
    }
    // CURL failed.
    return 0;
}