
#include "GRT_CUDA_host/GRTCandidateHashesSHA256.h"


// SHA256 will always have length 20 hashes.
GRTCandidateHashesSHA256::GRTCandidateHashesSHA256() : GRTCandidateHashes(32) {

}

// Copy all the various data into the GPU with the needed transfer function
void GRTCandidateHashesSHA256::copyDataToConstant(GRTThreadRunData *data) {
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


    copySHA256CandidateDataToConstant(hostCharset, charsetLength,
        this->TableHeader->getChainLength(), this->TableHeader->getTableIndex(),
        numberThreads);

    for (i = 0; i < 16; i++) {
        delete[] hostCharset2D[i];
    }
    delete[] hostCharset2D;

    delete[] CharsetLengths;

    return;
}

void GRTCandidateHashesSHA256::setHashInConstant(unsigned char *hash) {
    copySHA256HashDataToConstant(hash);
}

void GRTCandidateHashesSHA256::runCandidateHashKernel(int PasswordLength, int CUDA_Blocks, int CUDA_Threads,
        unsigned char *DEVICE_End_Hashes, UINT4 ThreadSpaceOffset, UINT4 StartStep, UINT4 StepsToRun) {
    LaunchSHA256CandidateHashKernel(PasswordLength, CUDA_Blocks, CUDA_Threads,
        DEVICE_End_Hashes, ThreadSpaceOffset, StartStep, StepsToRun);
}