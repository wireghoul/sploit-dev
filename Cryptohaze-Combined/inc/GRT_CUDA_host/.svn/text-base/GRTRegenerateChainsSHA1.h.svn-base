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

#ifndef __GRTREGENERATECHAINSSHA1_H__
#define __GRTREGENERATECHAINSSHA1_H__

#include "GRT_CUDA_host/GRTRegenerateChains.h"

extern "C" void copySHA1RegenerateDataToConstant(char *hostCharset, UINT4 hostCharsetLength,
        UINT4 hostChainLength, UINT4 hostTableIndex, UINT4 hostNumberOfThreads, unsigned char *hostBitmap,
        UINT4 hostNumberOfHashes);
extern "C" void setSHA1RegenerateNumberOfChains(UINT4 numberOfChains);

extern "C" void LaunchSHA1RegenerateKernel(int PasswordLength, int CUDA_Blocks, int CUDA_Threads,
        unsigned char *InitialPasswordArray, unsigned char *FoundPasswordArray,
        unsigned char *DeviceHashArray, UINT4 PasswordSpaceOffset, UINT4 StartChainIndex,
        UINT4 StepsToRun, UINT4 charset_offset, unsigned char *successArray, UINT4 hostNumberOfHashes);


class GRTRegenerateChainsSHA1 : public GRTRegenerateChains {
public:
    GRTRegenerateChainsSHA1();



protected:
    void copyDataToConstant(GRTRegenerateThreadRunData *data);
    void setNumberOfChainsToRegen(uint32_t);


    void Launch_CUDA_Kernel(unsigned char *InitialPasswordArray, unsigned char *FoundPasswordArray,
        unsigned char *DeviceHashArray, UINT4 PasswordSpaceOffset, UINT4 StartChainIndex,
        UINT4 StepsToRun, UINT4 charset_offset, unsigned char *successArray, GRTRegenerateThreadRunData *data);
};

#endif