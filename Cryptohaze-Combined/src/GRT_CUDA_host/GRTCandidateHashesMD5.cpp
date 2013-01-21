
#include "GRT_CUDA_host/GRTCandidateHashesMD5.h"

// MD5 will always have length 16 hashes.
GRTCandidateHashesMD5::GRTCandidateHashesMD5() : GRTCandidateHashes(16) {
    
}

// Copy all the various data into the GPU with the needed transfer function
void GRTCandidateHashesMD5::copyDataToConstant(GRTThreadRunData *data) {
    char hostCharset[512]; // The 512 byte array copied to the GPU
    int i;
    char** hostCharset2D; // The 16x256 array of characters
    uint32_t charsetLength;
    char *CharsetLengths;
    uint32_t numberThreads;

    hostCharset2D = this->TableHeader->getCharset();
    CharsetLengths = this->TableHeader->getCharsetLengths();
    numberThreads = this->ThreadData[data->threadID].CUDABlocks *
            this->ThreadData[data->threadID].CUDAThreads;

    charsetLength = CharsetLengths[0];

    //printf("Charset length: %d\n", charsetLength);

    for (i = 0; i < 512; i++) {
        hostCharset[i] = hostCharset2D[0][i % charsetLength];
    }


    copyMD5CandidateDataToConstant(hostCharset, charsetLength,
        this->TableHeader->getChainLength(), this->TableHeader->getTableIndex(),
        numberThreads);

    for (i = 0; i < 16; i++) {
        delete[] hostCharset2D[i];
    }
    delete[] hostCharset2D;

    delete[] CharsetLengths;

    return;
}

void GRTCandidateHashesMD5::setHashInConstant(unsigned char *hash) {
    copyMD5HashDataToConstant(hash);
    //printf ("Hash %02x%02x%02x... copied to constant.\n", hash[0], hash[1], hash[2]);
}

void GRTCandidateHashesMD5::runCandidateHashKernel(int PasswordLength, int CUDA_Blocks, int CUDA_Threads,
        unsigned char *DEVICE_End_Hashes, UINT4 ThreadSpaceOffset, UINT4 StartStep, UINT4 StepsToRun) {
    LaunchMD5CandidateHashKernel(PasswordLength, CUDA_Blocks, CUDA_Threads,
        DEVICE_End_Hashes, ThreadSpaceOffset, StartStep, StepsToRun);
}