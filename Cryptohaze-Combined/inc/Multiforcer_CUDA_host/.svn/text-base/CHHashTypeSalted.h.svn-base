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

// Extensions of the plain hash file type for salted hashes.

#ifndef __CHHASHTYPESALTED_H
#define __CHHASHTYPESALTED_H

#include "Multiforcer_CUDA_host/CHHashTypePlain.h"

// Forward declare needed classes
class CHHashFileSalted32;

class CHHashTypeSalted : public CHHashTypePlain {
public:
    // hash size, salt size
    CHHashTypeSalted(int newHashSize, int newSaltSize);

    void setHashFile(CHHashFileTypes *NewHashFile);

    void crackPasswordLength(int passwordLength);

protected:

    // Salt data
    int SaltLength;
    unsigned char *SaltList;
    unsigned char *SaltLengths;


    CHHashFileSalted32 *HashFile;

    // Store the salts and salt lengths in global memory to allow unlimited
    // salt size.
    unsigned char *DEVICE_Salts[MAX_SUPPORTED_THREADS];
    unsigned char *DEVICE_Salt_Lengths[MAX_SUPPORTED_THREADS];


    int outputFoundHashes(struct threadRunData *data);

    virtual void copyDataToConstant(char *hostCharset, int charsetLength,
        unsigned char *hostCharsetLengths, unsigned char *hostSharedBitmap, int threadId) = 0;

    virtual void Launch_CUDA_Kernel(int passlength, uint64_t charsetLength, int numberOfPasswords, unsigned char *DEVICE_Passwords,
            unsigned char *DEVICE_Success, struct start_positions *DEVICE_Start_Positions,
        uint64_t per_step, uint64_t threads, uint64_t blocks, unsigned char *DEVICE_Hashes,
        unsigned char *DEVICE_Bitmap) = 0;

    // Allocate memory on the host and GPU as needed.
    // This just adds the salt allocations and frees them when done.
//    virtual void GPU_Thread_Allocate_Memory(threadRunData *data);
    // Free all memory on the host and GPU for the thread.
//    virtual void GPU_Thread_Free_Memory(threadRunData *data);


};

#endif