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

#ifndef __GRT_CANDIDATEHASHES_MD5_H__
#define __GRT_CANDIDATEHASHES_MD5_H__

#include "GRT_CUDA_host/GRTCandidateHashes.h"

typedef uint32_t UINT4;

extern "C" void copyMD5CandidateDataToConstant(char *hostCharset, UINT4 hostCharsetLength,
        UINT4 hostChainLength, UINT4 hostTableIndex, UINT4 hostNumberOfThreads);

extern "C" void copyMD5HashDataToConstant(unsigned char *hash);

extern "C" void LaunchMD5CandidateHashKernel(int PasswordLength, int CUDA_Blocks, int CUDA_Threads,
        unsigned char *DEVICE_End_Hashes, UINT4 ThreadSpaceOffset, UINT4 StartStep, UINT4 StepsToRun);


class GRTCandidateHashesMD5 : public GRTCandidateHashes {
public:
    GRTCandidateHashesMD5();
    void copyDataToConstant(GRTThreadRunData *data);
    void runCandidateHashKernel(int PasswordLength, int CUDA_Blocks, int CUDA_Threads,
        unsigned char *DEVICE_End_Hashes, UINT4 ThreadSpaceOffset, UINT4 StartStep, UINT4 StepsToRun);
    void setHashInConstant(unsigned char *hash);
};

#endif
