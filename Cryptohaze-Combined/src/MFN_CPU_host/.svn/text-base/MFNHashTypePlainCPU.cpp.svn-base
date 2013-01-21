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

#include "MFN_CPU_host/MFNHashTypePlainCPU.h"
#include "MFN_Common/MFNCommandLineData.h"
#include "MFN_Common/MFNMultiforcerClassFactory.h"
#include "CH_HashFiles/CHHashFileVPlain.h"
#include "MFN_Common/MFNDebugging.h"
#include "CH_Common/CHHiresTimer.h"
#include "MFN_Common/MFNWorkunitBase.h"
#include "MFN_Common/MFNDefines.h"
#include "MFN_Common/MFNDisplay.h"

extern MFNClassFactory MultiforcerGlobalClassFactory;
extern struct global_commands global_interface;

MFNHashTypePlainCPU::MFNHashTypePlainCPU(int hashLengthBytes) :  MFNHashTypePlain(hashLengthBytes) {
    trace_printf("MFNHashTypePlainCPU::MFNHashTypePlainCPU(%d)\n", hashLengthBytes);

    this->MFNHashTypeMutex.lock();
    this->threadId = MultiforcerGlobalClassFactory.getDisplayClass()->getFreeThreadId(CPU_THREAD);
    this->numberThreads++;
    trace_printf("MFNHashType CPU Thread ID %d\n", this->threadId);
    this->MFNHashTypeMutex.unlock();
}

void MFNHashTypePlainCPU::setupDevice() {
    trace_printf("MFNHashTypePlainCPU::setupDevice()\n");
    // TODO: Let this size be set by the command line parameters.
    // Default of 1MB is good for high end CPUs, less-good for things like an Atom.
    this->createArbitraryBitmap(0, this->activeHashesProcessed, this->classBitmapLookup_a, 1024*1024);
}


void MFNHashTypePlainCPU::teardownDevice() {
    trace_printf("MFNHashTypePlainCPU::teardownDevice()\n");
}

void MFNHashTypePlainCPU::allocateThreadAndDeviceMemory() {
    trace_printf("MFNHashTypePlainCPU::allocateThreadAndDeviceMemory()\n");
    
    this->HostStartPointAddress = new uint8_t [this->TotalKernelWidth * this->passwordLength];
}


void MFNHashTypePlainCPU::freeThreadAndDeviceMemory() {
    trace_printf("MFNHashTypePlainCPU::freeThreadAndDeviceMemory()\n");
    
    delete[] this->HostStartPointAddress;
}


void MFNHashTypePlainCPU::copyDataToDevice() {
    trace_printf("MFNHashTypePlainCPU::copyDataToDevice()\n");
}

void MFNHashTypePlainCPU::copyStartPointsToDevice() {
    trace_printf("MFNHashTypePlainCPU::copyStartPointsToDevice()\n");
}


int MFNHashTypePlainCPU::setCPUThreads(int numberCpuThreads) {
    trace_printf("MFNHashTypePlainCPU::setCPUThreads(%d)\n", numberCpuThreads);
        
    this->numberCPUThreads = numberCpuThreads;
    
    // Resize the threading vectors.
    this->CPUThreadData.resize(this->numberCPUThreads);
    this->CPUThreadObjects.resize(this->numberCPUThreads);
    
    
    /**
     * All the CPU threads are considered to be a single kernel for the purposes
     * of this cracking type.  
     */
    this->TotalKernelWidth = this->numberCPUThreads * this->VectorWidth;
    
    trace_printf("Successfully added %d CPU threads\n", numberCpuThreads);
    trace_printf("Thread %d cores/totalwidth: %d/%d\n", this->threadId, 
            this->numberCPUThreads, this->TotalKernelWidth);
    
    return 1;
}

void MFNHashTypePlainCPU::setupClassForMultithreadedEntry() {
    trace_printf("MFNHashTypePlainCPU::setupClassForMultithreadedEntry()\n");
}

void MFNHashTypePlainCPU::synchronizeThreads() {
}


void MFNHashTypePlainCPU::setStartPoints(uint64_t perThread, uint64_t startPoint) {
    trace_printf("MFNHashTypePlain::setStartPoints()\n");

    uint32_t numberThreads = this->TotalKernelWidth;
    uint64_t threadId, threadStartPoint;
    uint32_t characterPosition;

    uint8_t *threadStartPosition = this->HostStartPointAddress;

    if (this->isSingleCharset) {
        klaunch_printf("Calculating start points for a single charset.\n");
        // Copy the current charset length into a local variable for speed.
        uint8_t currentCharsetLength = this->currentCharset.at(0).size();

        for (threadId = 0; threadId < numberThreads; threadId++) {
            threadStartPoint = threadId * perThread + startPoint;
            //printf("Thread %u, startpoint %lu\n", threadId, threadStartPoint);

            // Loop through all the character positions.  This is easier than a case statement.
            for (characterPosition = 0; characterPosition < this->passwordLength; characterPosition++) {
                threadStartPosition[characterPosition * numberThreads + threadId] =
                        (uint8_t)(threadStartPoint % currentCharsetLength);
                threadStartPoint /= currentCharsetLength;
                /*printf("Set thread %d to startpoint %d at pos %d\n",
                        threadId, threadStartPosition[characterPosition * numberThreads + threadId],
                        characterPosition * numberThreads + threadId);*/
            }
        }

    } else{
        klaunch_printf("Calculating start points for a multiple charset.\n");
        if (this->passwordLength > this->currentCharset.size()) {
            printf("Error: Password length > charset length!\n");
            printf("Terminating!\n");
            exit(1);
        }
        for (threadId = 0; threadId < numberThreads; threadId++) {
            threadStartPoint = threadId * perThread + startPoint;
            //printf("Thread %u, startpoint %lu\n", threadId, threadStartPoint);

            // Loop through all the character positions.  This is easier than a case statement.
            for (characterPosition = 0; characterPosition < this->passwordLength; characterPosition++) {
                threadStartPosition[characterPosition * numberThreads + threadId] =
                        (uint8_t)(threadStartPoint % this->currentCharset[characterPosition].size());
                threadStartPoint /= this->currentCharset[characterPosition].size();
                /*printf("Set thread %d to startpoint %d at pos %d\n",
                        threadId, threadStartPosition[characterPosition * numberThreads + threadId],
                        characterPosition * numberThreads + threadId);*/
            }
        }
    }
}


void MFNHashTypePlainCPU::copyDeviceFoundPasswordsToHost() {
    trace_printf("MFNHashTypePlainCPU::copyDeviceFoundPasswordsToHost()\n");
}

void MFNHashTypePlainCPU::outputFoundHashes() {
    trace_printf("MFNHashTypePlain::outputFoundHashes()\n");
}


void MFNHashTypePlainCPU::RunGPUWorkunit(struct MFNWorkunitRobustElement *WU) {
    trace_printf("MFNHashTypePlainCPU::RunGPUWorkunit()\n");

    /**
     * High-res timer - this should work properly on both Windows & Posix.
     */
    CHHiresTimer Timer, TotalTaskTimer;

    uint64_t perThread, start_point = 0;
    uint16_t thread;

    // Counter for the barrier waits.
    uint32_t barrierWaitLoop;
    
    CPUSSEThreadData threadData;

    /**
     * Create a barrier for synchronizing the threads and reporting performance
     * regularly.  This is numThreads + 1, for the calling thread.  The threads
     * will be blocked every so often, performance reported, and the threads
     * will continue.  This should not affect performance much at all.
     */
    boost::barrier SSEThreadBarrier(this->numberCPUThreads + 1);

    klaunch_printf("Thread %d total kernel width: %d\n", this->threadId, this->TotalKernelWidth);
    klaunch_printf("Thread %d blocks/threads/vec: %d/%d/%d\n", this->threadId, this->GPUBlocks, this->GPUThreads, this->VectorWidth);
    
    // Calculate how many iterations per thread - divide total by the number of
    // total threads, then add one to deal with truncation.
    perThread = WU->EndPoint - WU->StartPoint;
    perThread /= (this->TotalKernelWidth);
    perThread++;
    
    klaunch_printf("Total kernel width: %d\n", this->TotalKernelWidth);
    klaunch_printf("perThread: %d\n", perThread);


    // Set up the start points for all the threads.  This is only done once.
    this->setStartPasswords32(perThread, start_point + WU->StartPoint);
    
    
    threadData.classBarrier = &SSEThreadBarrier;
    threadData.numberStepsTotal = perThread;


    // Start the timers.
    Timer.start();
    TotalTaskTimer.start();

    for (thread = 0; thread < this->numberCPUThreads; thread++) {
        threadData.threadNumber = thread;
        this->CPUThreadData[thread] = threadData;
        this->CPUThreadObjects[thread] = new boost::thread(&MFNHashTypePlainCPU::cpuSSEThread, this, this->CPUThreadData[thread]);
    }

    // If there are more steps than the wait period, we will be inserting
    // barriers into the process to get timing information.
    if (perThread > MFNHASHTYPECPUPLAIN_STEPS_PER_PERIOD) {
        for (barrierWaitLoop = 0;
             barrierWaitLoop < (perThread / MFNHASHTYPECPUPLAIN_STEPS_PER_PERIOD);
             barrierWaitLoop++) {

            // Wait for all the threads to reach this point.
            //printf("Host thread waiting on barrier.\n");
            SSEThreadBarrier.wait();
            //printf("Host thread past barrier.\n");
            Timer.stop();
            klaunch_printf("Seconds: %f\n", Timer.getElapsedTimeInSec());
            klaunch_printf("Number hashes: %d\n", MFNHASHTYPECPUPLAIN_STEPS_PER_PERIOD * this->TotalKernelWidth);
            klaunch_printf("Kernel width: %d\n", this->TotalKernelWidth);
            klaunch_printf("Rate: %f /sec\n", 
                (MFNHASHTYPECPUPLAIN_STEPS_PER_PERIOD * this->TotalKernelWidth) /
                Timer.getElapsedTimeInSec());
            this->Display->setThreadCrackSpeed(this->threadId, 
                    (MFNHASHTYPECPUPLAIN_STEPS_PER_PERIOD * this->TotalKernelWidth) /
                    Timer.getElapsedTimeInSec());
            // If an exit has been requested, do so.
            if (global_interface.exit) {
                break;
            }
            Timer.start();
        }
    }

    // Join the threads - this will wait until they are completed.
    for (thread = 0; thread < this->numberCPUThreads; thread++) {
        this->CPUThreadObjects[thread]->join();
    }
    Timer.stop();
    TotalTaskTimer.stop();

    klaunch_printf("%lu hashes in %f s\n", (WU->EndPoint - WU->StartPoint), TotalTaskTimer.getElapsedTimeInSec());
    klaunch_printf("Rate: %f /sec\n", (WU->EndPoint - WU->StartPoint) / TotalTaskTimer.getElapsedTimeInSec());

    return;
}
