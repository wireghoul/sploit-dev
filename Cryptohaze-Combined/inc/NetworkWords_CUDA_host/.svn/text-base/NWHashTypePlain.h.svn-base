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

#ifndef __NWHASHTYPEPLAIN_H
#define __NWHASHTYPEPLAIN_H

/*
 * NWHashTypePlain handles "plain" hash types - MD5, NTLM, SHA1, etc.
 * It takes the size of the hash in bytes as the argument to it's
 * constructor.  This class should handle pretty much everything of the
 * plain hash type, with the only subfunctions needed being the hash specific
 * stuff that calls into the CUDA kernels.
 */

#include "NWHashType.h"
#include "CHCommon.h"

class NWHashTypePlain : public NWHashType {
public:
    // Constructor that takes the hash size in bytes and sets up the
    // needed structures to handle this.
    NWHashTypePlain(int hashBytes);

    // Entry back into the class
    virtual void GPU_Thread(void *pointer);

    // Runs the crack for a given length.  Spawns the threads/etc.
    virtual void crackPasswords();

protected:


#if USE_BOOST_THREADS
    boost::mutex mutex3Boost;
#else
    pthread_mutex_t mutex3;  // Mutex for important things.
    pthread_mutexattr_t mutex3attr;
#endif

    // The length of this hash type, in bytes.
    // MD5/NTLM = 16, SHA1 = 20, SHA256 = 32, etc.
    int hashLengthBytes;

    // Hash data - constant across all threads.
    unsigned char *HashList;
    uint64_t NumberOfHashes;

    // Host and device array pointers for each thread.
    // This stores the pointer to the device hash list for each device.
    unsigned char *DEVICE_Hashes[MAX_SUPPORTED_THREADS];
    
    unsigned char *HOST_Passwords[MAX_SUPPORTED_THREADS];
    unsigned char *DEVICE_Passwords[MAX_SUPPORTED_THREADS];

    unsigned char *HOST_Success[MAX_SUPPORTED_THREADS];
    unsigned char *HOST_Success_Reported[MAX_SUPPORTED_THREADS];
    unsigned char *DEVICE_Success[MAX_SUPPORTED_THREADS];

    unsigned char *DEVICE_512MB_Bitmap[MAX_SUPPORTED_THREADS];

    // Keep the per-step count to prevent slowdowns
    uint64_t per_step[MAX_SUPPORTED_THREADS];

    // Bitmaps for lookup - constant across all threads.
    unsigned char hostConstantBitmap[8192];
    unsigned char *hostBitmap512MB;

    // Functions to create the bitmaps
    virtual void createConstantBitmap8kb();
    virtual void createGlobalBitmap512MB();


    // Allocate memory on the host and GPU as needed.
    virtual void GPU_Thread_Allocate_Memory(threadRunData *data);
    // Free all memory on the host and GPU for the thread.
    virtual void GPU_Thread_Free_Memory(threadRunData *data);

    virtual int outputFoundHashes(struct threadRunData *data);
    // Copies the data to whatever charset is needed.
    // Implemented in the hash type file.
    virtual void copyDataToConstant(unsigned char *hostSharedBitmap, int threadId) = 0;

    // Launches the appropriate CUDA kernel
    virtual void Launch_CUDA_Kernel(unsigned char *Wordlist, uint32_t numberOfWords,
            int numberOfPasswords, unsigned char *DEVICE_Passwords,
            unsigned char *DEVICE_Success, struct start_positions *DEVICE_Start_Positions,
        uint64_t per_step, uint64_t threads, uint64_t blocks, unsigned char *DEVICE_Hashes,
        unsigned char *DEVICE_Bitmap) = 0;

};

#endif