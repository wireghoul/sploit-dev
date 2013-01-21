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

#ifndef _GRTGENERATETABLE_H
#define _GRTGENERATETABLE_H

#include "GRT_CUDA_host/GRTGenCommandLineData.h"
#include "GRT_Common/GRTCharsetSingle.h"
#include "CUDA_Common/CUDA_Common.h"
#include "CH_Common/CHRandom.h"
#include "GRT_Common/GRTTableHeader.h"

// Password types: NORMAL is a normally packed password,
// NTLM is the utf16le encoding
#define PASSWORD_TYPE_NORMAL 0
#define PASSWORD_TYPE_NTLM 1

typedef uint32_t UINT4;

// This is a generic class for all hash types.  Tweak as needed.
class GRTGenerateTable {
protected:
    // The length of the hash output block.  Should be a multiple of 4.
    int HashOutputBlockLengthBytes;
    // The length of the password input block.  Should be a multiple of 4.
    int PasswordInputBlockLengthBytes;
    unsigned char *HOST_Initial_Passwords, *DEVICE_Initial_Passwords;
    unsigned char *HOST_End_Hashes, *DEVICE_End_Hashes;

    GRTGenCommandLineData *TableParameters;
    GRTCharsetSingle *Charset;
    CHRandom *RandomGenerator;
    // This is here for WebGen - so we can reference it and modify as needed.
    GRTTableHeader *CurrentTableHeader;


    //===== Functions that are mostly the same ===
    // These functions are basically the same for all hash types.

    // Generates the initial passwords.
    // Specify the type to generate: NORMAL or NTLM
    virtual void generateInitialPasswords(int passwordType);

    // Allocates space for the host and device hashes.
    // Also sets the device being used.
    virtual void mallocDeviceAndHostSpace();
    // And undo it.
    virtual void freeDeviceAndHostSpace();

    //===== Per Hash Functions ====
    // These need to be implemented by the specific hash type being used.

    // Copy the constant values to the GPU.  This is per-hash-implementation specific.
    virtual void copyConstantsToGPU(char *HOST_Charset, UINT4 HOST_Charset_Length,
        UINT4 HOST_Chain_Length, UINT4 HOST_Number_Of_Chains, UINT4 HOST_Table_Index,
        UINT4 HOST_Number_Of_Threads) = 0;

    // Invokes the kernel.
    virtual void runKernel(int passwordLength, UINT4 CUDA_Blocks,
        UINT4 CUDA_Threads, unsigned char *DEVICE_Initial_Passwords,
        unsigned char *DEVICE_End_Hashes, UINT4 PasswordSpaceOffset,
        UINT4 CurrentChainStartOffset, UINT4 StepsPerInvocation, UINT4 CharsetOffset) = 0;

    // Uploads the webgen file to the webserver.
    int uploadWebGenTableFile(char *filename);


public:
    GRTGenerateTable(int hashLengthBytes, int passwordLengthBytes);
    ~GRTGenerateTable();
    // Actually create the tables.
    virtual void createTables();
    //  Set the table parameter structure to the created one.
    virtual void setGRTGenCommandLineData(GRTGenCommandLineData *NewTableParameters);
    virtual void setGRTCharsetSingle(GRTCharsetSingle *NewCharset);
    
    virtual void setRandomGenerator(CHRandom *newRandomGenerator) {
        this->RandomGenerator = newRandomGenerator;
    }
};

#endif
