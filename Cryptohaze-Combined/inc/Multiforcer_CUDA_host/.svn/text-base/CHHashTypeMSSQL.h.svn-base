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

#ifndef __CHHASHTYPEMSSQL_H
#define __CHHASHTYPEMSSQL_H

#include "Multiforcer_CUDA_host/CHHashTypePlain.h"

class CHHashFileMSSQL;

extern "C" void copyMSSQLDataToConstant(char *hostCharset, int charsetLength,
        unsigned char *hostCharsetLengths, unsigned char *hostSharedBitmap, 
        int threadId, uint32_t *salts);


extern "C" void Launch_CUDA_MSSQL_Kernel(int passlength, uint64_t charsetLength, int numberOfPasswords, unsigned char *DEVICE_Passwords,
            unsigned char *DEVICE_Success, struct start_positions *DEVICE_Start_Positions,
        uint64_t per_step, uint64_t threads, uint64_t blocks, unsigned char *DEVICE_Hashes,
        unsigned char *DEVICE_Bitmap);

class CHHashTypeMSSQL : public CHHashTypePlain {
public:
    CHHashTypeMSSQL();

    // Need to modify this slightly
    void crackPasswordLength(int passwordLength);
    // Add the CHHashFileMSSQL type to handle the extra fields needed
    void setHashFile(CHHashFileTypes *NewHashFile);

protected:

    // Salt data
    uint32_t *SaltList;

    CHHashFileMSSQL *HashFile;

    int outputFoundHashes(struct threadRunData *data);

    void copyDataToConstant(char *hostCharset, int charsetLength,
        unsigned char *hostCharsetLengths, unsigned char *hostSharedBitmap, int threadId);

    void Launch_CUDA_Kernel(int passlength, uint64_t charsetLength, int numberOfPasswords, unsigned char *DEVICE_Passwords,
            unsigned char *DEVICE_Success, struct start_positions *DEVICE_Start_Positions,
        uint64_t per_step, uint64_t threads, uint64_t blocks, unsigned char *DEVICE_Hashes,
        unsigned char *DEVICE_Bitmap);
};

#endif