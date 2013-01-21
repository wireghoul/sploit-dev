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

#ifndef __CHHASHTYPEPLAIN_H
#define __CHHASHTYPEPLAIN_H

/*
 * CHHashTypePlain handles "plain" hash types - MD5, NTLM, SHA1, etc.
 * It takes the size of the hash in bytes as the argument to it's
 * constructor.  This class should handle pretty much everything of the
 * plain hash type, with the only subfunctions needed being the hash specific
 * stuff that calls into the CUDA kernels.
 */

#include "Multiforcer_CUDA_host/CHHashType.h"
#include "Multiforcer_Common/CHCommon.h"

struct CHWorkunitElement;


class CHHashTypePlain : public CHHashType {
public:
    // Constructor that takes the hash size in bytes and sets up the
    // needed structures to handle this.
    CHHashTypePlain(int hashBytes);

    // Entry back into the class
    virtual void GPU_Thread(void *pointer);

    // Runs the crack for a given length.  Spawns the threads/etc.
    virtual void crackPasswordLength(int passwordLength);

	// Runs a GPU workunit.
    virtual void RunGPUWorkunit(struct CHWorkunitRobustElement *WU, struct threadRunData *data);
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

    // The length of the current password we are working on.
    int passwordLength;
    
    // Set to 1 if we need all thread to exit due to a null workunit received.
    // Trying to get password length changing working...
    char threadRendezvous; 

    // Hash data - constant across all threads.
    unsigned char *HashList;
    uint64_t NumberOfHashes;

    // Charset data to the GPUs.
    // This gets put in constant memory by the per-hash copyToConstant call.
    char *hostConstantCharset;
    uint32_t hostConstantCharsetLength;
    unsigned char hostCharsetLengths[MAX_PASSWORD_LEN];

    // Host and device array pointers for each thread.
    // This stores the pointer to the device hash list for each device.
    unsigned char *DEVICE_Hashes[MAX_SUPPORTED_THREADS];
    
    unsigned char *HOST_Passwords[MAX_SUPPORTED_THREADS];
    unsigned char *DEVICE_Passwords[MAX_SUPPORTED_THREADS];

    unsigned char *HOST_Success[MAX_SUPPORTED_THREADS];
    unsigned char *HOST_Success_Reported[MAX_SUPPORTED_THREADS];
    unsigned char *DEVICE_Success[MAX_SUPPORTED_THREADS];

    unsigned char *DEVICE_512MB_Bitmap[MAX_SUPPORTED_THREADS];

    // Start points for the threads
    unsigned char *HOST_Start_Points[MAX_SUPPORTED_THREADS];
    unsigned char *DEVICE_Start_Points[MAX_SUPPORTED_THREADS];

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

    virtual void setStartPointsMulti(start_positions *HOST_Start_Positions, uint64_t perThread, uint64_t numberOfThreads,
				    unsigned char *charset_length_array, uint64_t start_point, int passLength);

    // Just uses the charset in position 0.
    virtual void setStartPointsSingle(start_positions *HOST_Start_Positions, uint64_t perThread, uint64_t numberOfThreads,
				    uint64_t charsetLength, uint64_t start_point, int passLength);

    virtual int outputFoundHashes(struct threadRunData *data);
    
    // Copies the data to whatever charset is needed.
    // Implemented in the hash type file.
    virtual void copyDataToConstant(char *hostCharset, int charsetLength,
        unsigned char *hostCharsetLengths, unsigned char *hostSharedBitmap, int threadId) = 0;

    // Launches the appropriate CUDA kernel
    virtual void Launch_CUDA_Kernel(int passlength, uint64_t charsetLength, int numberOfPasswords, unsigned char *DEVICE_Passwords,
            unsigned char *DEVICE_Success, struct start_positions *DEVICE_Start_Positions,
        uint64_t per_step, uint64_t threads, uint64_t blocks, unsigned char *DEVICE_Hashes,
        unsigned char *DEVICE_Bitmap) = 0;

};

#endif