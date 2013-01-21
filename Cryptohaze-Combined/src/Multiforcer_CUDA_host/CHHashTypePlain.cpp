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

#include "Multiforcer_CUDA_host/CHHashTypePlain.h"

#include "Multiforcer_Common/CHDisplay.h"
#include "Multiforcer_CUDA_host/CHCommandLineData.h"
#include "CH_Common/CHHashFileTypes.h"
#include "CH_Common/CHCharset.h"
#include "CH_Common/CHWorkunitRobust.h"
#include "CH_Common/CHWorkunitNetwork.h"
#include "Multiforcer_Common/CHCommon.h"


// Ugly stuff to let pthreads work.
// We need to go to boost threads here at some point...
extern "C" {
    void *CHHashTypeGPUThread(void *);
}

void *CHHashTypeGPUThread(void * pointer) {
    struct threadRunData *data;

    data = (threadRunData *) pointer;

    //printf("IN THREAD %d\n", data->threadID);
    data->HashType->GPU_Thread(pointer);
    //printf("Thread %d Back from GPU_Thread\n", data->threadID);
    fflush(stdout);
#if !USE_BOOST_THREADS
    pthread_exit(NULL);
#else
	return NULL;
#endif
}

extern struct global_commands global_interface;


CHHashTypePlain::CHHashTypePlain(int hashBytes) {
    this->hashLengthBytes = hashBytes;

    this->hostBitmap512MB = NULL;
    this->HashList = NULL;

    // Clear out all the arrays before we do anything else.
    memset(this->DEVICE_Hashes, 0, sizeof(unsigned char *) * MAX_SUPPORTED_THREADS);

    memset(this->HOST_Passwords, 0, sizeof(unsigned char *) * MAX_SUPPORTED_THREADS);
    memset(this->DEVICE_Passwords, 0, sizeof(unsigned char *) * MAX_SUPPORTED_THREADS);

    memset(this->HOST_Success, 0, sizeof(unsigned char *) * MAX_SUPPORTED_THREADS);
    memset(this->HOST_Success_Reported, 0, sizeof(unsigned char *) * MAX_SUPPORTED_THREADS);
    memset(this->DEVICE_Success, 0, sizeof(unsigned char *) * MAX_SUPPORTED_THREADS);

    memset(this->HOST_Start_Points, 0, sizeof(unsigned char *) * MAX_SUPPORTED_THREADS);
    memset(this->DEVICE_Start_Points, 0, sizeof(unsigned char *) * MAX_SUPPORTED_THREADS);

    memset(this->DEVICE_512MB_Bitmap, 0, sizeof(unsigned char *) * MAX_SUPPORTED_THREADS);

    memset(this->per_step, 0, sizeof(uint64_t) * MAX_SUPPORTED_THREADS);
#if !USE_BOOST_THREADS
    pthread_mutexattr_init(&this->mutex3attr);
    pthread_mutex_init(&this->mutex3, &this->mutex3attr);
#endif
}



void CHHashTypePlain::createConstantBitmap8kb() {
    // Initialize the 8kb shared memory bitmap.
    // 8192 bytes, 8 bits per byte.
    // Bits are determined by the first 16 bits of a hash.
    // Byte index is 13 bits, bit index is 3 bits.
    // Algorithm: To set:
    // First 13 bits of the hash (high order bits) are used as the
    // index to the array.
    // Next 3 bits control the left-shift amount of the '1' bit.
    // So, hash 0x0000 has byte 0, LSB set.
    // Hash 0x0105 has byte
    uint64_t i;
    unsigned char bitmap_byte;
    uint32_t bitmap_index;
    // Zero bitmap
    memset(this->hostConstantBitmap, 0, 8192);

    for (i = 0; i < this->NumberOfHashes; i++) {
        // Load first two bytes of hash - reversed order to work with the little endian memory storage.  Otherwise the GPU has to flip things around
        // and this is operations that should not have to be performed.
        bitmap_index = (this->HashList[i * this->hashLengthBytes + 1] << 8) + this->HashList[(i * this->hashLengthBytes)];
        //printf("Hash %lu: 2 bytes: %02X %02X ", i, this->ActiveHashList[i * 16], this->ActiveHashList[i * 16 + 1]);
        //printf("Bitmap index: %04X\n", bitmap_index);
        // Shift left by the lowest 3 bits
        bitmap_byte = 0x01 << (bitmap_index & 0x0007);
        bitmap_index = bitmap_index >> 3;
        //printf(" Index %u, byte %02X\n", bitmap_index, bitmap_byte);
        this->hostConstantBitmap[bitmap_index] |= bitmap_byte;
    }
    if (false) {
        printf("Bitmap output\n");
        for (i = 0; i < 8192; i++) {
            if (i % 4 == 0) {
                printf("\n Index %llu: ", i);
            }
            printf("%02X ", this->hostConstantBitmap[i]);
        }
    }
}


void CHHashTypePlain::createGlobalBitmap512MB() {
    // Does the same thing as above, but for a much larger bitmap for global memory space.
    // Right now, this only works for 512MB hash space (32 bits worth of bitmap).
    // Else returns a NULL pointer & does not do any work.

    uint32_t i;
    unsigned char bitmap_byte;
    uint32_t bitmap_index;
    // Allocate bitmap.  If already allocated, skip it.
    if (!this->hostBitmap512MB) {
        sprintf(this->statusBuffer, "Alloc 512MB bitmap\n");
        this->Display->addStatusLine(this->statusBuffer);
        this->hostBitmap512MB = (unsigned char *)new (std::nothrow) unsigned char[512*1024*1024];
    }
    if (this->hostBitmap512MB == NULL) {
        sprintf(this->statusBuffer, "Unable to allocate 512MB host RAM - something has gone very wrong.  Exiting.\n");
        this->Display->addStatusLine(this->statusBuffer);
        exit(1);
    }
    memset(this->hostBitmap512MB, 0, 512*1024*1024);



    for (i = 0; i < this->NumberOfHashes; i++) {
        // Load first 4 bytes of hash.  This is loaded normally,
        bitmap_index = (this->HashList[i * this->hashLengthBytes]) | (this->HashList[(i * this->hashLengthBytes + 1)] << 8)
                | (this->HashList[(i * this->hashLengthBytes + 2)] << 16) | (this->HashList[(i * this->hashLengthBytes + 3)] << 24);
        //printf("Hash %ld: 4 bytes: %02X %02X %02X %02X", i, Bitmap_Hashes->hashes[i * Bitmap_Hashes->hash_length],
            //Bitmap_Hashes->hashes[i * Bitmap_Hashes->hash_length + 1], Bitmap_Hashes->hashes[i * Bitmap_Hashes->hash_length + 2],
            //Bitmap_Hashes->hashes[i * Bitmap_Hashes->hash_length + 3]);
        //printf("Bitmap index: %08X\n", bitmap_index);
        // Shift left by the lowest 3 bits
        bitmap_byte = 0x01 << (bitmap_index & 0x0007);
        bitmap_index = bitmap_index >> 3;
        //printf(" Index %ld, byte %02X\n", bitmap_index, bitmap_byte);
        this->hostBitmap512MB[bitmap_index] |= bitmap_byte;
    }
}


void CHHashTypePlain::GPU_Thread_Allocate_Memory(threadRunData *data) {
    // Malloc space on the GPU for everything.
    cudaError_t err;

    // Default flags are for memory on device and copying things.
    unsigned int flags = 0;

    // If we are using zero copy, set the zero copy flag.
    if (this->CommandLineData->GetUseZeroCopy()) {
        flags = cudaHostAllocMapped;
    }
    // Malloc device hash space.
    //TODO: Replace CUDA_SAFE_CALL with custom handler
    CUDA_SAFE_CALL(cudaMalloc((void **)&this->DEVICE_Hashes[data->threadID],
        this->NumberOfHashes * this->hashLengthBytes * sizeof(unsigned char)));
    CUDA_SAFE_CALL(cudaMemcpy(this->DEVICE_Hashes[data->threadID], this->HashList,
        this->NumberOfHashes * this->hashLengthBytes * sizeof(unsigned char), cudaMemcpyHostToDevice));

    //this->HOST_Success[data->threadID] = new unsigned char [this->NumberOfHashes * sizeof(unsigned char)];
    cudaHostAlloc((void **)&this->HOST_Success[data->threadID],
        this->NumberOfHashes * sizeof(unsigned char), flags);
    memset(this->HOST_Success[data->threadID], 0, this->NumberOfHashes * sizeof(unsigned char));
    this->HOST_Success_Reported[data->threadID] = new unsigned char [this->NumberOfHashes * sizeof(unsigned char)];
    memset(this->HOST_Success_Reported[data->threadID], 0, this->NumberOfHashes * sizeof(unsigned char));

    // If zero copy is in use, get the device pointer
    if (this->CommandLineData->GetUseZeroCopy()) {
        cudaHostGetDevicePointer((void **)&this->DEVICE_Success[data->threadID],
            this->HOST_Success[data->threadID], 0);
    } else {
        CUDA_SAFE_CALL(cudaMalloc((void **)&this->DEVICE_Success[data->threadID],
            this->NumberOfHashes * sizeof(unsigned char)));
        CUDA_SAFE_CALL(cudaMemset(this->DEVICE_Success[data->threadID], 0,
            this->NumberOfHashes * sizeof(unsigned char)));
    }

    //this->HOST_Passwords[data->threadID] = new unsigned char[MAX_PASSWORD_LEN * this->NumberOfHashes * sizeof(unsigned char)];
    cudaHostAlloc((void **)&this->HOST_Passwords[data->threadID], 
        MAX_PASSWORD_LEN * this->NumberOfHashes * sizeof(unsigned char), flags);
    memset(this->HOST_Passwords[data->threadID], 0, MAX_PASSWORD_LEN * this->NumberOfHashes * sizeof(unsigned char));

    if (this->CommandLineData->GetUseZeroCopy()) {
        cudaHostGetDevicePointer((void **)&this->DEVICE_Passwords[data->threadID],
            this->HOST_Passwords[data->threadID], 0);
    } else {
        CUDA_SAFE_CALL(cudaMalloc((void **)&this->DEVICE_Passwords[data->threadID],
            MAX_PASSWORD_LEN * this->NumberOfHashes * sizeof(unsigned char)));
        CUDA_SAFE_CALL(cudaMemset(this->DEVICE_Passwords[data->threadID], 0,
            MAX_PASSWORD_LEN * this->NumberOfHashes * sizeof(unsigned char)));
    }
    // Host and device start positions
    // Write combined as this will not be read by the host.
    CUDA_SAFE_CALL(cudaHostAlloc((void**)&this->HOST_Start_Points[data->threadID],
        data->CUDABlocks * data->CUDAThreads * sizeof(struct start_positions),
        cudaHostAllocWriteCombined | flags));

    // Check the last cudaMalloc for failure
    CUDA_SAFE_CALL(cudaMalloc((void **)&this->DEVICE_Start_Points[data->threadID],
        data->CUDABlocks * data->CUDAThreads * sizeof(struct start_positions)));
    cudaMemset(this->DEVICE_Start_Points[data->threadID], 0, data->CUDABlocks * data->CUDAThreads * sizeof(struct start_positions));


    // If the 512MB bitmap is being used, set it up and copy it.
    if (this->CommandLineData->GetUseLookupTable() || getCudaIsFermi(data->gpuDeviceId)) {
        CUDA_SAFE_CALL(cudaMalloc((void **)&this->DEVICE_512MB_Bitmap[data->threadID],
            512 * 1024 * 1024 * sizeof(unsigned char)));
        CUDA_SAFE_CALL(cudaMemcpy(this->DEVICE_512MB_Bitmap[data->threadID], this->hostBitmap512MB,
            512 * 1024 * 1024 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d: CUDA error 5: %s. Exiting.\n",
                data->threadID, cudaGetErrorString( err));
        return;
    }
#if USE_BOOST_THREADS
    this->mutex3Boost.lock();
#else
    pthread_mutex_lock(&this->mutex3);
#endif
    sprintf(this->statusBuffer, "Thread %d mem loaded\n", data->threadID);
    this->Display->addStatusLine(this->statusBuffer);
#if USE_BOOST_THREADS
    this->mutex3Boost.unlock();
#else
    pthread_mutex_unlock(&this->mutex3);
#endif
}

void CHHashTypePlain::GPU_Thread_Free_Memory(threadRunData *data) {
    // Malloc space on the GPU for everything.
    cudaError_t err;
    // Malloc device hash space.
    //TODO: Replace CUDA_SAFE_CALL with custom handler
    CUDA_SAFE_CALL(cudaFree(this->DEVICE_Hashes[data->threadID]));

    //delete[] this->HOST_Success[data->threadID];
    cudaFreeHost((void *)this->HOST_Success[data->threadID]);

    delete[] this->HOST_Success_Reported[data->threadID];

    if (!this->CommandLineData->GetUseZeroCopy()) {
        CUDA_SAFE_CALL(cudaFree(this->DEVICE_Success[data->threadID]));
    }

    //delete[] this->HOST_Passwords[data->threadID];
    cudaFreeHost((void *)this->HOST_Passwords[data->threadID]);
    if (!this->CommandLineData->GetUseZeroCopy()) {
        CUDA_SAFE_CALL(cudaFree(this->DEVICE_Passwords[data->threadID]));
    }
    // Host and device start positions
    //delete[] this->HOST_Start_Points[data->threadID];
    cudaFreeHost((void*)this->HOST_Start_Points[data->threadID]);
    // Check the last cudaMalloc for failure
    CUDA_SAFE_CALL(cudaFree(this->DEVICE_Start_Points[data->threadID]));

    // If the 512MB bitmap is being used, set it up and copy it.
    if (this->CommandLineData->GetUseLookupTable() || getCudaIsFermi(data->gpuDeviceId)) {
        CUDA_SAFE_CALL(cudaFree(this->DEVICE_512MB_Bitmap[data->threadID]));
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Thread %d: CUDA error 5: %s. Exiting.\n",
                data->threadID, cudaGetErrorString( err));
        return;
    }
}


// Do the main work of this type.
void CHHashTypePlain::crackPasswordLength(int passwordLength) {
    int i;
    
    this->threadRendezvous = 0;
    
    this->Display->setPasswordLen(passwordLength);
    this->passwordLength = passwordLength;

    // Do the global work
	this->HashList = this->HashFile->ExportUncrackedHashList();
    this->NumberOfHashes = this->HashFile->GetUncrackedHashCount();
    this->hostConstantCharset = this->Charset->getCharset();
    this->hostConstantCharsetLength = this->Charset->getCharsetNumberElements();

    this->Display->setTotalHashes(this->NumberOfHashes);

    for (i = 0; i < passwordLength; i++) {
        this->hostCharsetLengths[i] = this->Charset->getCharsetLength(i);
    }

    // If this is *not* a server-only instance, do stuff.
    if (!this->CommandLineData->GetIsServerOnly()) {
        this->createConstantBitmap8kb();
        // Go ahead & create the global bitmap anyway.
        this->createGlobalBitmap512MB();
    }
    
#if USE_BOOST_THREADS
    memset(this->ThreadObjects, 0, MAX_SUPPORTED_THREADS * sizeof(boost::thread *));
#else
    memset(this->ThreadIds, 0, MAX_SUPPORTED_THREADS * sizeof(pthread_t));
#endif

    // All global work done.  Onto the per-thread work.
    sprintf(this->statusBuffer, "Creating %d threads\n", this->ActiveThreadCount);
    this->Display->addStatusLine(this->statusBuffer);
    // Enter all the threads
    for(i = 0; i < this->ActiveThreadCount; i++) {
#if USE_BOOST_THREADS
        this->ThreadObjects[i] = new boost::thread(&CHHashTypeGPUThread, &this->ThreadData[i]);
#else
        pthread_create(&this->ThreadIds[i], NULL, CHHashTypeGPUThread, &this->ThreadData[i] );
#endif
    }
    // Wait for them to come back.
    for(i = 0; i < this->ActiveThreadCount; i++) {
#if USE_BOOST_THREADS
        this->ThreadObjects[i]->join();
#else
        pthread_join(this->ThreadIds[i], NULL);
#endif
    }
    sprintf(this->statusBuffer, "Threads joined\n");
    this->Display->addStatusLine(this->statusBuffer);

    // Wait until all workunits are back from remote systems.
    if (this->Workunit->GetNumberOfCompletedWorkunits() < this->Workunit->GetNumberOfWorkunits()) {
        sprintf(this->statusBuffer, "Waiting for workunits...\n");
        this->Display->addStatusLine(this->statusBuffer);
    }

    while (this->Workunit->GetNumberOfCompletedWorkunits() < this->Workunit->GetNumberOfWorkunits()) {
        CHSleep(1);
        this->Display->Refresh();
        // Make termination work properly for the server
        if (global_interface.exit) {
            break;
        }
    }
    if (this->hostBitmap512MB) {
        delete[] this->hostBitmap512MB;
        this->hostBitmap512MB = NULL;
    }
}



// New version to go from zero up, instead of down.
void CHHashTypePlain::setStartPointsMulti(start_positions *HOST_Start_Positions,
uint64_t perThread, uint64_t numberOfThreads,  unsigned char *charset_length_array,
uint64_t start_point, int passLength) {

    uint64_t thread, thread_start;
    for (thread = 0; thread < numberOfThreads; thread++) {
	// Where in the password space the thread starts: ThreadID * number per thread + current start offset
        thread_start = thread * perThread + start_point;
        //printf("Thread %d: Start point %d\n", thread, thread_start);
	// Start at the right point for the password length and fall through to the end

        //Needed for the single hash reversing stuff.  Implement this sanely.
        HOST_Start_Positions[thread].p0 = (unsigned char)(thread_start % charset_length_array[0]);
        thread_start /= charset_length_array[0];
        if (passLength == 1) {
            continue;
        }
        HOST_Start_Positions[thread].p1 = (unsigned char)(thread_start % charset_length_array[1]);
        thread_start /= charset_length_array[1];
        if (passLength == 2) {
            continue;
        }
        HOST_Start_Positions[thread].p2 = (unsigned char)(thread_start % charset_length_array[2]);
        thread_start /= charset_length_array[2];
        if (passLength == 3) {
            continue;
        }
        HOST_Start_Positions[thread].p3 = (unsigned char)(thread_start % charset_length_array[3]);
        thread_start /= charset_length_array[3];
        if (passLength == 4) {
            continue;
        }
        HOST_Start_Positions[thread].p4 = (unsigned char)(thread_start % charset_length_array[4]);
        thread_start /= charset_length_array[4];
        if (passLength == 5) {
            continue;
        }
        HOST_Start_Positions[thread].p5 = (unsigned char)(thread_start % charset_length_array[5]);
        thread_start /= charset_length_array[5];
        if (passLength == 6) {
            continue;
        }
        HOST_Start_Positions[thread].p6 = (unsigned char)(thread_start % charset_length_array[6]);
        thread_start /= charset_length_array[6];
        if (passLength == 7) {
            continue;
        }
        HOST_Start_Positions[thread].p7 = (unsigned char)(thread_start % charset_length_array[7]);
        thread_start /= charset_length_array[7];
        if (passLength == 8) {
            continue;
        }
        HOST_Start_Positions[thread].p8 = (unsigned char)(thread_start % charset_length_array[8]);
        thread_start /= charset_length_array[8];
        if (passLength == 9) {
            continue;
        }
        HOST_Start_Positions[thread].p9 = (unsigned char)(thread_start % charset_length_array[9]);
        thread_start /= charset_length_array[9];
        if (passLength == 10) {
            continue;
        }
        HOST_Start_Positions[thread].p10 = (unsigned char)(thread_start % charset_length_array[10]);
        thread_start /= charset_length_array[10];
        if (passLength == 11) {
            continue;
        }
        HOST_Start_Positions[thread].p11 = (unsigned char)(thread_start % charset_length_array[11]);
        thread_start /= charset_length_array[11];
        if (passLength == 12) {
            continue;
        }
        HOST_Start_Positions[thread].p12 = (unsigned char)(thread_start % charset_length_array[12]);
        thread_start /= charset_length_array[12];
        if (passLength == 13) {
            continue;
        }
        HOST_Start_Positions[thread].p13 = (unsigned char)(thread_start % charset_length_array[13]);
        thread_start /= charset_length_array[13];
        if (passLength == 14) {
            continue;
        }
        HOST_Start_Positions[thread].p14 = (unsigned char)(thread_start % charset_length_array[14]);
        thread_start /= charset_length_array[14];
        if (passLength == 15) {
            continue;
        }
        HOST_Start_Positions[thread].p15 = (unsigned char)(thread_start % charset_length_array[15]);
        thread_start /= charset_length_array[15];
        if (passLength == 16) {
            continue;
        }
        HOST_Start_Positions[thread].p16 = (unsigned char)(thread_start % charset_length_array[16]);
        thread_start /= charset_length_array[16];
        if (passLength == 17) {
            continue;
        }
        HOST_Start_Positions[thread].p17 = (unsigned char)(thread_start % charset_length_array[17]);
        thread_start /= charset_length_array[17];
        if (passLength == 18) {
            continue;
        }
        HOST_Start_Positions[thread].p18 = (unsigned char)(thread_start % charset_length_array[18]);
        thread_start /= charset_length_array[18];
        if (passLength == 19) {
            continue;
        }
        HOST_Start_Positions[thread].p19 = (unsigned char)(thread_start % charset_length_array[19]);
        thread_start /= charset_length_array[19];
        if (passLength == 20) {
            continue;
        }
        HOST_Start_Positions[thread].p20 = (unsigned char)(thread_start % charset_length_array[20]);
        thread_start /= charset_length_array[20];
        if (passLength == 21) {
            continue;
        }
        HOST_Start_Positions[thread].p21 = (unsigned char)(thread_start % charset_length_array[21]);
        thread_start /= charset_length_array[21];
        if (passLength == 22) {
            continue;
        }
        HOST_Start_Positions[thread].p22 = (unsigned char)(thread_start % charset_length_array[22]);
        thread_start /= charset_length_array[22];
        if (passLength == 23) {
            continue;
        }
        HOST_Start_Positions[thread].p23 = (unsigned char)(thread_start % charset_length_array[23]);
        thread_start /= charset_length_array[23];
        if (passLength == 24) {
            continue;
        }
        HOST_Start_Positions[thread].p24 = (unsigned char)(thread_start % charset_length_array[24]);
        thread_start /= charset_length_array[24];
        if (passLength == 25) {
            continue;
        }
        HOST_Start_Positions[thread].p25 = (unsigned char)(thread_start % charset_length_array[25]);
        thread_start /= charset_length_array[25];
        if (passLength == 26) {
            continue;
        }
        HOST_Start_Positions[thread].p26 = (unsigned char)(thread_start % charset_length_array[26]);
        thread_start /= charset_length_array[26];
        if (passLength == 27) {
            continue;
        }
        HOST_Start_Positions[thread].p27 = (unsigned char)(thread_start % charset_length_array[27]);
        thread_start /= charset_length_array[27];
        if (passLength == 28) {
            continue;
        }
        HOST_Start_Positions[thread].p28 = (unsigned char)(thread_start % charset_length_array[28]);
        thread_start /= charset_length_array[28];
        if (passLength == 29) {
            continue;
        }
        HOST_Start_Positions[thread].p29 = (unsigned char)(thread_start % charset_length_array[29]);
        thread_start /= charset_length_array[29];
        if (passLength == 30) {
            continue;
        }
        HOST_Start_Positions[thread].p30 = (unsigned char)(thread_start % charset_length_array[30]);
        thread_start /= charset_length_array[30];
        if (passLength == 31) {
            continue;
        }
        HOST_Start_Positions[thread].p31 = (unsigned char)(thread_start % charset_length_array[31]);
        thread_start /= charset_length_array[31];
        if (passLength == 32) {
            continue;
        }
        HOST_Start_Positions[thread].p32 = (unsigned char)(thread_start % charset_length_array[32]);
        thread_start /= charset_length_array[32];
        if (passLength == 33) {
            continue;
        }
        HOST_Start_Positions[thread].p33 = (unsigned char)(thread_start % charset_length_array[33]);
        thread_start /= charset_length_array[33];
        if (passLength == 34) {
            continue;
        }
        HOST_Start_Positions[thread].p34 = (unsigned char)(thread_start % charset_length_array[34]);
        thread_start /= charset_length_array[34];
        if (passLength == 35) {
            continue;
        }
        HOST_Start_Positions[thread].p35 = (unsigned char)(thread_start % charset_length_array[35]);
        thread_start /= charset_length_array[35];
        if (passLength == 36) {
            continue;
        }
        HOST_Start_Positions[thread].p36 = (unsigned char)(thread_start % charset_length_array[36]);
        thread_start /= charset_length_array[36];
        if (passLength == 37) {
            continue;
        }
        HOST_Start_Positions[thread].p37 = (unsigned char)(thread_start % charset_length_array[37]);
        thread_start /= charset_length_array[37];
        if (passLength == 38) {
            continue;
        }
        HOST_Start_Positions[thread].p38 = (unsigned char)(thread_start % charset_length_array[38]);
        thread_start /= charset_length_array[38];
        if (passLength == 39) {
            continue;
        }
        HOST_Start_Positions[thread].p39 = (unsigned char)(thread_start % charset_length_array[39]);
        thread_start /= charset_length_array[39];
        if (passLength == 40) {
            continue;
        }
        HOST_Start_Positions[thread].p40 = (unsigned char)(thread_start % charset_length_array[40]);
        thread_start /= charset_length_array[40];
        if (passLength == 41) {
            continue;
        }
        HOST_Start_Positions[thread].p41 = (unsigned char)(thread_start % charset_length_array[41]);
        thread_start /= charset_length_array[41];
        if (passLength == 42) {
            continue;
        }
        HOST_Start_Positions[thread].p42 = (unsigned char)(thread_start % charset_length_array[42]);
        thread_start /= charset_length_array[42];
        if (passLength == 43) {
            continue;
        }
        HOST_Start_Positions[thread].p43 = (unsigned char)(thread_start % charset_length_array[43]);
        thread_start /= charset_length_array[43];
        if (passLength == 44) {
            continue;
        }
        HOST_Start_Positions[thread].p44 = (unsigned char)(thread_start % charset_length_array[44]);
        thread_start /= charset_length_array[44];
        if (passLength == 45) {
            continue;
        }
        HOST_Start_Positions[thread].p45 = (unsigned char)(thread_start % charset_length_array[45]);
        thread_start /= charset_length_array[45];
        if (passLength == 46) {
            continue;
        }
        HOST_Start_Positions[thread].p46 = (unsigned char)(thread_start % charset_length_array[46]);
        thread_start /= charset_length_array[46];
        if (passLength == 47) {
            continue;
        }
        HOST_Start_Positions[thread].p47 = (unsigned char)(thread_start % charset_length_array[47]);
        thread_start /= charset_length_array[47];
        if (passLength == 48) {
            continue;
        }

    }
}

void CHHashTypePlain::setStartPointsSingle(start_positions *HOST_Start_Positions, uint64_t perThread, uint64_t numberOfThreads,
				    uint64_t charsetLength, uint64_t start_point, int passLength) {

    uint64_t thread, thread_start;
	for (thread = 0; thread < numberOfThreads; thread++) {
	  // Where in the password space the thread starts: ThreadID * number per thread + current start offset
      thread_start = thread * perThread + start_point;
	  // Start at the right point for the password length and fall through to the end
      switch (passLength) {
	    case 16:
	      HOST_Start_Positions[thread].p15 = (unsigned char)(thread_start % charsetLength);
          thread_start /= charsetLength;
	    case 15:
	      HOST_Start_Positions[thread].p14 = (unsigned char)(thread_start % charsetLength);
          thread_start /= charsetLength;
	    case 14:
	      HOST_Start_Positions[thread].p13 = (unsigned char)(thread_start % charsetLength);
          thread_start /= charsetLength;
	    case 13:
	      HOST_Start_Positions[thread].p12 = (unsigned char)(thread_start % charsetLength);
          thread_start /= charsetLength;
	    case 12:
	      HOST_Start_Positions[thread].p11 = (unsigned char)(thread_start % charsetLength);
          thread_start /= charsetLength;
	    case 11:
	      HOST_Start_Positions[thread].p10 = (unsigned char)(thread_start % charsetLength);
          thread_start /= charsetLength;
	    case 10:
	      HOST_Start_Positions[thread].p9 = (unsigned char)(thread_start % charsetLength);
          thread_start /= charsetLength;
	    case 9:
	      HOST_Start_Positions[thread].p8 = (unsigned char)(thread_start % charsetLength);
          thread_start /= charsetLength;
	    case 8:
	      HOST_Start_Positions[thread].p7 = (unsigned char)(thread_start % charsetLength);
          thread_start /= charsetLength;
	    case 7:
	      HOST_Start_Positions[thread].p6 = (unsigned char)(thread_start % charsetLength);
          thread_start /= charsetLength;
	    case 6:
	      HOST_Start_Positions[thread].p5 = (unsigned char)(thread_start % charsetLength);
          thread_start /= charsetLength;
	    case 5:
	      HOST_Start_Positions[thread].p4 = (unsigned char)(thread_start % charsetLength);
          thread_start /= charsetLength;
	    case 4:
	      HOST_Start_Positions[thread].p3 = (unsigned char)(thread_start % charsetLength);
          thread_start /= charsetLength;
            case 3:
              HOST_Start_Positions[thread].p2 = (unsigned char)(thread_start % charsetLength);
          thread_start /= charsetLength;
           case 2:
              HOST_Start_Positions[thread].p1 = (unsigned char)(thread_start % charsetLength);
              thread_start /= charsetLength;
           case 1:
              HOST_Start_Positions[thread].p0 = (unsigned char)(thread_start % charsetLength);
	  }
    }
}


/* outputFoundHashes takes the various arrays, looks for newly found passwords, prints them, and outputs them to the output file if needed.
 * It returns the number of passwords it found that were new.
 */
int CHHashTypePlain::outputFoundHashes(struct threadRunData *data) {
  int i, j;
  int passwordsFound = 0;
    for (i = 0; i < this->NumberOfHashes; i++) {
        if (this->HOST_Success[data->threadID][i] && !this->HOST_Success_Reported[data->threadID][i]) {
            this->HashFile->ReportFoundPassword(&this->HashList[i * this->hashLengthBytes],
                    &this->HOST_Passwords[data->threadID][MAX_PASSWORD_LEN * i]);
            passwordsFound++;
            this->HOST_Success_Reported[data->threadID][i] = 1;

            for (j = 0; j < strlen((const char *)&this->HOST_Passwords[data->threadID][MAX_PASSWORD_LEN * i]); j++) {
                this->statusBuffer[j] = this->HOST_Passwords[data->threadID][MAX_PASSWORD_LEN * i + j];
            }
            this->statusBuffer[j] = 0;
            this->Display->addCrackedPassword(this->statusBuffer);
        }
    }
  this->Display->addCrackedHashes(passwordsFound);

  // Check to see if we should exit (as all hashes are found).
  if (this->HashFile->GetUncrackedHashCount() == 0) {
      global_interface.exit = 1;
  }

  return passwordsFound;
}

// This is the GPU thread where we do the per-GPU tasks.
void CHHashTypePlain::GPU_Thread(void *pointer) {
    struct threadRunData *data;
    struct CHWorkunitRobustElement WU;

    // Per-thread client ID from workunit class
    uint16_t ClientId = 0;

    data = (threadRunData *) pointer;

    // CUDA device 0 is the fastest in the system.
    // If we are NOT device 0, sleep for a second to make sure the fastest
    // GPU will grab work if there is only one workunit.
    // Ordering beyond that is a don't-care condition.
    if (data->gpuDeviceId) {
        CHSleep(1);
    }

    // Set the device.
    cudaSetDevice(data->gpuDeviceId);
    // Enable blocking sync.  This dramatically reduces CPU usage.
    // If zero copy is being used, set DeviceMapHost as well
    if (this->CommandLineData->GetUseZeroCopy()) {
        cudaSetDeviceFlags(cudaDeviceBlockingSync | cudaDeviceMapHost);
    } else {
        cudaSetDeviceFlags(cudaDeviceBlockingSync);
    }
    
    // We now are in the context of this device for all future CUDA calls.
    // Reset the per-step data for the new password length.
    this->per_step[data->threadID] = 0;
    GPU_Thread_Allocate_Memory(data);


    this->copyDataToConstant(this->hostConstantCharset, this->hostConstantCharsetLength,
        this->hostCharsetLengths, this->hostConstantBitmap, data->threadID);
    cudaThreadSynchronize();

    // Get our per-thread client ID
    ClientId = this->Workunit->GetClientId();
    sprintf(this->statusBuffer, "Td %d: CID %d.\n", data->threadID, ClientId);
    this->Display->addStatusLine(this->statusBuffer);

    // I... *think* we're ready to rock!
    // As long as we aren't supposed to exit, keep running.
    while(!global_interface.exit && !this->threadRendezvous) {
        WU = this->Workunit->GetNextWorkunit(ClientId);
        if (!WU.IsValid) {
            // If a null workunit came in, rendezvous the threads.
            this->threadRendezvous = 1;
            // Workunit came back null -
#if USE_BOOST_THREADS
            this->mutex3Boost.lock();
#else
            pthread_mutex_lock(&this->mutex3);
#endif
            sprintf(this->statusBuffer, "Td %d: out of WU.\n", data->threadID);
            this->Display->addStatusLine(this->statusBuffer);
#if USE_BOOST_THREADS
            this->mutex3Boost.unlock();
#else
            pthread_mutex_unlock(&this->mutex3);
#endif
            break;
        } /*else if (WU == WORKUNIT_PLZ_HOLD) {
            // While we are being told to hold, sleep for a second then try again.
            while (WU == WORKUNIT_PLZ_HOLD) {
                CHSleep(1);
                WU = this->Workunit->GetNextWorkunit(ClientId);
            }
        }*/
        if (this->CommandLineData->GetDevDebug()) {
            printf("Thread %d has workunit ID %lld\n", data->threadID, WU.WorkUnitID);
        }
        this->RunGPUWorkunit(&WU, data);

        // If we are NOT aborting, submit the unit.
        // If we are force-exiting, do not submit the workunit!
        if (!global_interface.exit) {
            this->Workunit->SubmitWorkunit(WU);
        }
        this->Display->setWorkunitsCompleted(this->Workunit->GetNumberOfCompletedWorkunits());
        //sprintf(this->statusBuffer, "WU rate: %0.1f", this->Workunit->GetAverageRate());
        //this->Display->addStatusLine(this->statusBuffer);

    }
    GPU_Thread_Free_Memory(data);
    // Free all thread resources to eliminate warnings with CUDA 4.0
    cudaThreadExit();
}


void CHHashTypePlain::RunGPUWorkunit(struct CHWorkunitRobustElement *WU, struct threadRunData *data) {

    //unsigned int timer, timer_step;
    cudaEvent_t timer_start;
    cudaEvent_t timer_stop;
    float ref_time = 0.0f;

    // These are all uint64_t to prevent underflow issues.
    // They must remain 64-bit or things break.
    //uint64_t blocks, threads;
    uint64_t perThread, start_point = 0;
    uint64_t step_count = 0;
    //uint64_t per_step;

    // Default number per steps.  As this is updated quickly, this just needs to be in the ballpark.
    if (this->per_step[data->threadID] == 0) {
        this->per_step[data->threadID] = 5;
    }

    // Calculate how many iterations per thread.
    perThread = WU->EndPoint - WU->StartPoint;
    perThread /= (data->CUDABlocks * data->CUDAThreads);
    perThread++;

    // Threadsafe timers...
    cudaEventCreate(&timer_start);
    cudaEventCreate(&timer_stop);

    cudaEventRecord(timer_start, 0);

    while (start_point <= perThread) {
        step_count++;

        this->setStartPointsMulti((start_positions*)this->HOST_Start_Points[data->threadID],
                perThread, (data->CUDABlocks * data->CUDAThreads), this->hostCharsetLengths,
                start_point + WU->StartPoint, this->passwordLength);

        // We sync here and wait for the GPU to finish.
        cudaThreadSynchronize();

        cudaEventRecord(timer_stop, 0);
        cudaEventSynchronize(timer_stop);


        // Wait for it to stabilize slightly before getting timer values, to handle large target_ms values
        /*if (step_count > 2) {
            // Get ref_time
            cudaEventElapsedTime(&ref_time, timer_start, timer_stop);
        } else {
            ref_time = 0;
        }*/
        cudaEventElapsedTime(&ref_time, timer_start, timer_stop);

        // Run this roughly every second, or every step if target_ms is >500
        if ((step_count < 5) || (data->kernelTimeMs > 500) || (step_count % (1000 / data->kernelTimeMs) == 0)) {
            // Copy device success & password lists to the host, look for new successes, and print them out "on the fly" (and, optionally, to the output file)
            CUDA_SAFE_CALL(cudaMemcpy(this->HOST_Success[data->threadID], this->DEVICE_Success[data->threadID],
                    this->NumberOfHashes * sizeof (unsigned char), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(this->HOST_Passwords[data->threadID], this->DEVICE_Passwords[data->threadID],
                    MAX_PASSWORD_LEN * this->NumberOfHashes * sizeof (unsigned char), cudaMemcpyDeviceToHost));


            this->outputFoundHashes(data);
            // Only set the crack speed if we have set one...
            if (step_count > 5) {
                this->Display->setThreadCrackSpeed(data->threadID, 1, (float) (data->CUDABlocks * data->CUDAThreads * this->per_step[data->threadID]) / (ref_time * 1000.0));
            }
            // If the current execution time is not correct, adjust.
            if ((ref_time > 0) && (step_count > 2) &&
                    ((ref_time < (data->kernelTimeMs * 0.9)) ||
                    (ref_time > (data->kernelTimeMs * 1.1)))) {
                this->per_step[data->threadID] = (uint64_t) ((float) this->per_step[data->threadID] * ((float) data->kernelTimeMs / ref_time));
                if (0) {
                    printf("\nThread %d Adjusting passwords per step to %d\n", data->threadID, (unsigned int) this->per_step[data->threadID]);
                }
            }
        }

        // If we are to pause, hang here.
        if (global_interface.pause) {
            while (global_interface.pause) {
                // Just hang out until the pause is broken...
				CHSleep(1);
			}
        }
        // Exit option
        if (global_interface.exit) {
            return;
        }

        
        cudaMemcpy(this->DEVICE_Start_Points[data->threadID],
                this->HOST_Start_Points[data->threadID],
                data->CUDABlocks * data->CUDAThreads * sizeof (struct start_positions),
                cudaMemcpyHostToDevice);
        cudaThreadSynchronize();
        cudaEventRecord(timer_start, 0);
        Launch_CUDA_Kernel(this->passwordLength, 16, this->NumberOfHashes,
                this->DEVICE_Passwords[data->threadID], this->DEVICE_Success[data->threadID],
                (start_positions *)this->DEVICE_Start_Points[data->threadID], this->per_step[data->threadID],
                data->CUDAThreads, data->CUDABlocks, this->DEVICE_Hashes[data->threadID],
                this->DEVICE_512MB_Bitmap[data->threadID]);

        // Increment start point by however many we did
        start_point += this->per_step[data->threadID];

    }
    cudaThreadSynchronize();
    // Copy device success & password lists to the host, look for new successes, and print them out "on the fly" (and, optionally, to the output file)
    CUDA_SAFE_CALL(cudaMemcpy(this->HOST_Success[data->threadID], this->DEVICE_Success[data->threadID],
            this->NumberOfHashes * sizeof (unsigned char), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(this->HOST_Passwords[data->threadID], this->DEVICE_Passwords[data->threadID],
            MAX_PASSWORD_LEN * this->NumberOfHashes * sizeof (unsigned char), cudaMemcpyDeviceToHost));
    this->outputFoundHashes(data);
    cudaEventDestroy(timer_start);
    cudaEventDestroy(timer_stop);

    return;
}
