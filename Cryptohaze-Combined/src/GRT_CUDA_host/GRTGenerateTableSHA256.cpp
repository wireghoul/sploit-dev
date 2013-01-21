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

#include "GRT_CUDA_host/GRTGenerateTableSHA256.h"
#include <stdio.h>

extern "C" void copyConstantsToSHA256(char *HOST_Charset, UINT4 HOST_Charset_Length,
    UINT4 HOST_Chain_Length, UINT4 HOST_Number_Of_Chains, UINT4 HOST_Table_Index,
    UINT4 HOST_Number_Of_Threads);

extern "C" void LaunchGenerateKernelSHA256(int passwordLength, UINT4 CUDA_Blocks, UINT4 CUDA_Threads,
        unsigned char *DEVICE_Initial_Passwords,
    unsigned char *DEVICE_End_Hashes, UINT4 PasswordSpaceOffset, UINT4 CurrentChainStartOffset,
    UINT4 StepsPerInvocation, UINT4 CharsetOffset);


// Copy the constant values to the GPU.  This is per-hash-implementation specific.
void GRTGenerateTableSHA256::copyConstantsToGPU(char *HOST_Charset, UINT4 HOST_Charset_Length,
        UINT4 HOST_Chain_Length, UINT4 HOST_Number_Of_Chains, UINT4 HOST_Table_Index,
        UINT4 HOST_Number_Of_Threads) {

    copyConstantsToSHA256(HOST_Charset, HOST_Charset_Length,
    HOST_Chain_Length, HOST_Number_Of_Chains, HOST_Table_Index,
    HOST_Number_Of_Threads);
}

/**
 * Note: Even though SHA256 has more significant bits than SHA1, we're keeping
 * the SHA1 output length, as no more than 128 bits is being used, and there's
 * no point in keeping the extra data only to throw it away immediateyl.  This
 * could probably be reduced to 16 bytes without loss of efficiency.
 */
GRTGenerateTableSHA256::GRTGenerateTableSHA256() : GRTGenerateTable(20, 16) {
    return;
}

void GRTGenerateTableSHA256::runKernel(int passwordLength, UINT4 CUDA_Blocks,
        UINT4 CUDA_Threads, unsigned char *DEVICE_Initial_Passwords,
        unsigned char *DEVICE_End_Hashes, UINT4 PasswordSpaceOffset,
        UINT4 CurrentChainStartOffset, UINT4 StepsPerInvocation, UINT4 CharsetOffset) {
    LaunchGenerateKernelSHA256(passwordLength, CUDA_Blocks,
        CUDA_Threads, DEVICE_Initial_Passwords,
        DEVICE_End_Hashes, PasswordSpaceOffset,
        CurrentChainStartOffset, StepsPerInvocation, CharsetOffset);
}
