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


#include <vector>

//#define KERNEL_LAUNCH_PRINTF 1
//#define TRACE_PRINTF 1

#include "MFN_Common/MFNHashTypePlain.h"
#include "CH_HashFiles/CHHashFileV.h"
#include "CH_Common/CHCharsetNew.h"
#include "MFN_Common/MFNCommandLineData.h"
#include "MFN_Common/MFNWorkunitBase.h"
#include "CH_Common/CHHiresTimer.h"
#include "MFN_Common/MFNDefines.h"
#include "MFN_Common/MFNDebugging.h"
#include "MFN_Common/MFNDisplay.h"
#include "MFN_Common/MFNNetworkClient.h"
#include "MFN_Common/MFNMultiforcerClassFactory.h"

extern struct global_commands global_interface;

// static data members
uint8_t MFNHashTypePlain::staticDataInitialized = 0;
uint16_t MFNHashTypePlain::hashLengthBytes = 0;
uint16_t MFNHashTypePlain::passwordLength = 0;
uint16_t MFNHashTypePlain::passwordLengthWords = 0;
uint16_t MFNHashTypePlain::maxFoundPlainLength = 0;

boost::mutex MFNHashTypePlain::MFNHashTypePlainMutex;


std::vector<std::vector<uint8_t> > MFNHashTypePlain::activeHashesRaw;
std::vector<std::vector<uint8_t> > MFNHashTypePlain::activeHashesProcessed;
std::vector<uint8_t> MFNHashTypePlain::activeHashesProcessedDeviceformat;
std::vector<std::vector<uint8_t> > MFNHashTypePlain::currentCharset;

std::vector<uint8_t> MFNHashTypePlain::sharedBitmap4kb_a;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap4kb_b;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap4kb_c;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap4kb_d;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap4kb_e;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap4kb_f;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap4kb_g;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap4kb_h;

std::vector<uint8_t> MFNHashTypePlain::sharedBitmap8kb_a;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap8kb_b;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap8kb_c;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap8kb_d;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap8kb_e;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap8kb_f;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap8kb_g;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap8kb_h;

std::vector<uint8_t> MFNHashTypePlain::sharedBitmap16kb_a;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap16kb_b;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap16kb_c;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap16kb_d;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap16kb_e;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap16kb_f;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap16kb_g;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap16kb_h;

std::vector<uint8_t> MFNHashTypePlain::sharedBitmap32kb_a;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap32kb_b;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap32kb_c;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap32kb_d;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap32kb_e;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap32kb_f;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap32kb_g;
std::vector<uint8_t> MFNHashTypePlain::sharedBitmap32kb_h;

std::vector<uint8_t> MFNHashTypePlain::globalBitmap256kb_a;
std::vector<uint8_t> MFNHashTypePlain::globalBitmap256kb_b;
std::vector<uint8_t> MFNHashTypePlain::globalBitmap256kb_c;
std::vector<uint8_t> MFNHashTypePlain::globalBitmap256kb_d;
std::vector<uint8_t> MFNHashTypePlain::globalBitmap256kb_e;
std::vector<uint8_t> MFNHashTypePlain::globalBitmap256kb_f;
std::vector<uint8_t> MFNHashTypePlain::globalBitmap256kb_g;
std::vector<uint8_t> MFNHashTypePlain::globalBitmap256kb_h;

std::vector<uint8_t> MFNHashTypePlain::globalBitmap128mb_a;
std::vector<uint8_t> MFNHashTypePlain::globalBitmap128mb_b;
std::vector<uint8_t> MFNHashTypePlain::globalBitmap128mb_c;
std::vector<uint8_t> MFNHashTypePlain::globalBitmap128mb_d;
std::vector<uint8_t> MFNHashTypePlain::globalBitmap128mb_e;
std::vector<uint8_t> MFNHashTypePlain::globalBitmap128mb_f;
std::vector<uint8_t> MFNHashTypePlain::globalBitmap128mb_g;
std::vector<uint8_t> MFNHashTypePlain::globalBitmap128mb_h;

std::vector<uint8_t> MFNHashTypePlain::charsetForwardLookup;
std::vector<uint8_t> MFNHashTypePlain::charsetReverseLookup;
std::vector<uint8_t> MFNHashTypePlain::charsetLengths;


uint8_t MFNHashTypePlain::isSingleCharset;

CHHiresTimer MFNHashTypePlain::NetworkSpeedReportingTimer;

// Static data for salted hashes
boost::mutex MFNHashTypePlain::MFNHashTypeSaltedMutex;
std::vector<uint32_t> MFNHashTypePlain::saltLengths;
std::vector<std::vector<uint8_t> > MFNHashTypePlain::activeSalts;
std::vector<uint32_t> MFNHashTypePlain::activeSaltsDeviceformat;
uint8_t MFNHashTypePlain::saltedStaticDataInitialized;

std::vector<uint32_t> MFNHashTypePlain::activeIterationCounts;
std::vector<std::vector<uint8_t> > MFNHashTypePlain::otherData1;
std::vector<uint32_t> MFNHashTypePlain::otherData1Deviceformat;
std::vector<std::vector<uint8_t> > MFNHashTypePlain::otherData2;
std::vector<uint32_t> MFNHashTypePlain::otherData2Deviceformat;
std::vector<std::vector<uint8_t> > MFNHashTypePlain::otherData3;
std::vector<uint32_t> MFNHashTypePlain::otherData3Deviceformat;
std::vector<std::vector<uint8_t> > MFNHashTypePlain::otherData4;
std::vector<uint32_t> MFNHashTypePlain::otherData4Deviceformat;
std::vector<std::vector<uint8_t> > MFNHashTypePlain::otherData5;
std::vector<uint32_t> MFNHashTypePlain::otherData5Deviceformat;


extern MFNClassFactory MultiforcerGlobalClassFactory;

// Snagged from here:
// http://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c/
// Feb 23 2012
static char isPowerOfTwo (uint32_t x)
{
 uint32_t numberOfOneBits = 0;

 while(x && numberOfOneBits <=1)
   {
    if ((x & 1) == 1) /* Is the least significant bit a 1? */
      numberOfOneBits++;
    x >>= 1;          /* Shift number one bit to the right */
   }

 return (numberOfOneBits == 1); /* 'True' if only one 1 bit */
}

// Table search predicates.

/**
 * Table search predicate for big endian hashes.
 *
 * This function sorts a big endian hash (where the value, in the registers,
 * corresponds to the hash interpreted as big endian).  This is for SHA type hashes.
 *
 * Returns true if h1 less than h2
 */
static bool hashBigEndianSortPredicate(const std::vector<uint8_t> &h1, const std::vector<uint8_t> &h2) {
    uint32_t i;
    for (i = 0; i < h1.size(); i++) {
        if (h1[i] == h2[i]) {
            continue;
        } else if (h1[i] > h2[i]) {
            return 0;
        } else if (h1[i] < h2[i]) {
            return 1;
        }
    }
    // Exactly equal = return 0.
    return 0;
}

// Deal with hashes that are little endian 32-bits.
static bool hashLittleEndianSortPredicate(const std::vector<uint8_t> &h1, const std::vector<uint8_t> &h2) {
    long int i, j;
    
    for (i = 0; i < (h1.size() / 4); i++) {
        for (j = 3; j >= 0; j--) {
            if (h1[(i * 4) + j] == h2[(i * 4) + j]) {
                continue;
            } else if (h1[(i * 4) + j] > h2[(i * 4) + j]) {
                return 0;
            } else if (h1[(i * 4) + j] < h2[(i * 4) + j]) {
                return 1;
            }
        }
    }
    // Exactly equal = return 0.
    return 0;
}


bool hashUniquePredicate(const std::vector<uint8_t> &h1, const std::vector<uint8_t> &h2) {
    int i;
    for (i = 0; i < h1.size(); i++) {
        if (h1[i] != h2[i]) {
            return 0;
        }
    }
    // Exactly equal = return 1.
    return 1;
}


MFNHashTypePlain::MFNHashTypePlain(uint16_t newHashLengthBytes) : MFNHashType() {
    trace_printf("MFNHashTypePlain::MFNHashTypePlain(%d)\n", newHashLengthBytes);
    this->hashLengthBytes = newHashLengthBytes;
    
    // Clear the data on thread initalization.
    this->staticDataInitialized = 0;
    this->passwordLength = 0;

    this->activeHashesRaw.clear();
    this->activeHashesProcessed.clear();
    this->activeHashesProcessedDeviceformat.clear();
    this->currentCharset.clear();
    
    // Clear all the bitmaps - may have previous data.
    this->sharedBitmap4kb_a.clear();
    this->sharedBitmap4kb_b.clear();
    this->sharedBitmap4kb_c.clear();
    this->sharedBitmap4kb_d.clear();
    this->sharedBitmap4kb_e.clear();
    this->sharedBitmap4kb_f.clear();
    this->sharedBitmap4kb_g.clear();
    this->sharedBitmap4kb_h.clear();
    
    this->sharedBitmap8kb_a.clear();
    this->sharedBitmap8kb_b.clear();
    this->sharedBitmap8kb_c.clear();
    this->sharedBitmap8kb_d.clear();
    this->sharedBitmap8kb_e.clear();
    this->sharedBitmap8kb_f.clear();
    this->sharedBitmap8kb_g.clear();
    this->sharedBitmap8kb_h.clear();

    this->sharedBitmap16kb_a.clear();
    this->sharedBitmap16kb_b.clear();
    this->sharedBitmap16kb_c.clear();
    this->sharedBitmap16kb_d.clear();
    this->sharedBitmap16kb_e.clear();
    this->sharedBitmap16kb_f.clear();
    this->sharedBitmap16kb_g.clear();
    this->sharedBitmap16kb_h.clear();

    this->sharedBitmap32kb_a.clear();
    this->sharedBitmap32kb_b.clear();
    this->sharedBitmap32kb_c.clear();
    this->sharedBitmap32kb_d.clear();
    this->sharedBitmap32kb_e.clear();
    this->sharedBitmap32kb_f.clear();
    this->sharedBitmap32kb_g.clear();
    this->sharedBitmap32kb_h.clear();

    this->globalBitmap256kb_a.clear();
    this->globalBitmap256kb_b.clear();
    this->globalBitmap256kb_c.clear();
    this->globalBitmap256kb_d.clear();
    this->globalBitmap256kb_e.clear();
    this->globalBitmap256kb_f.clear();
    this->globalBitmap256kb_g.clear();
    this->globalBitmap256kb_h.clear();
    
    this->globalBitmap128mb_a.clear();
    this->globalBitmap128mb_b.clear();
    this->globalBitmap128mb_c.clear();
    this->globalBitmap128mb_d.clear();
    this->globalBitmap128mb_e.clear();
    this->globalBitmap128mb_f.clear();
    this->globalBitmap128mb_g.clear();
    this->globalBitmap128mb_h.clear();
    
    this->charsetForwardLookup.clear();
    this->charsetReverseLookup.clear();
    this->charsetLengths.clear();

    this->isSingleCharset = 0;

    this->threadRendezvous = 0;
    
    this->GPUBlocks = 0;
    this->GPUThreads = 0;
    this->VectorWidth = 1;
    this->TotalKernelWidth = 0;
    
    // Assume default case of unsalted hashes.
    this->numberUniqueSalts = 1;

    // Salted data is not set.
    this->saltedStaticDataInitialized = 0;
    
    // Clear the attribute array - classes must set this on their own.
    memset(&this->hashAttributes, 0, sizeof(this->hashAttributes));
}


void MFNHashTypePlain::crackPasswordLength(int passwordLength) {
    trace_printf("MFNHashTypePlain::crackPasswordLength(%d)\n", passwordLength);

    uint64_t i;
    char statusBuffer[1000];
    struct MFNWorkunitRobustElement WU;

    // New cracking - do NOT need to rendezvous threads.
    this->threadRendezvous = 0;

    // Acquire a setup mutex if we're the first thread.
    this->MFNHashTypePlainMutex.lock();

    // If static data is not set up, do so.
    // This data is shared across all instances.
    if (!this->staticDataInitialized) {
        this->threadRendezvous = 0;

        mt_printf("Thread %d doing MFNHashTypePlain setup.\n", this->threadId);
        
        // Handle the password length.  Set it in the display, copy it to the
        // internal buffer, and determine the max plain length to be supported.
        this->Display->setPasswordLen(passwordLength);
        this->passwordLength = passwordLength;
        this->setMaxFoundPlainLength();

        // Determine the password length in words.  If not a multiple of 4, round up.
        // Include the end padding bit (byte) in this calculation.
        this->passwordLengthWords = (this->passwordLength + 1);
        if (this->passwordLengthWords % 4) {
            this->passwordLengthWords = (this->passwordLengthWords + 4) & 0xfffc;
        }

        
        // Get the raw list of hashes.
        this->activeHashesRaw = this->HashFile->ExportUncrackedHashList();
        // Get the active charset.
        this->currentCharset = this->Charset->getCharset();
        
        mt_printf("Thread %d Charset length: %d\n", this->threadId, this->currentCharset.size());

        // If the charset length is 1, it is a single charset.  Tag as such.
        if (this->currentCharset.size() == 1) {
            this->isSingleCharset = 1;
        } else {
            this->isSingleCharset = 0;
        }
        
        if (this->CommandLineData->GetDevDebug()) {
            printf("=========Charset Dumping========\n");
            printf("isSingleCharset: %d\n", this->isSingleCharset);
            for (int pos = 0; pos < this->currentCharset.size(); pos++) {
                for (int chr = 0; chr < this->currentCharset[pos].size(); chr++) {
                    printf("%c", (char)this->currentCharset[pos][chr]);
                }
                printf("\n");
            }
        }
        
        this->setupCharsetArrays();

        this->Display->Refresh();

        // Preprocess all the hashes for the current password length.
        for (i = 0; i < this->activeHashesRaw.size(); i++) {
            this->activeHashesProcessed.push_back(this->preProcessHash(this->activeHashesRaw[i]));
        }

        // Sort and unique the hashes.
        this->sortHashes();

        for (i = 0; i < this->activeHashesProcessed.size(); i++) {
            static_printf("hash %d: ", i);
            for (int j = 0; j < this->activeHashesProcessed[i].size(); j++) {
                static_printf("%02x", this->activeHashesProcessed[i][j]);
            }
            static_printf("\n");
        }

        // Set up the device-format hash list
        this->copyHashesIntoDeviceFormat();

        // If this is *not* a server-only instance, create bitmaps
        if (!this->CommandLineData->GetIsServerOnly()) {
            this->createLookupBitmaps();
        }
        
        // Perform any additional setup needed.
        //this->doAdditionalStaticDataSetup();
        if (this->hashAttributes.hashUsesSalt) {
            this->MFNHashTypeSaltedMutex.lock();
            this->setupActiveSaltArrays();
            this->numberUniqueSalts = this->activeSalts.size();
            // Ensure there is no division by zero going on...
            if (this->numberUniqueSalts == 0) {
                this->numberUniqueSalts = 1;
            }
            this->MFNHashTypeSaltedMutex.unlock();
        }
        
        
        // Start the network reporting timer.
        this->NetworkSpeedReportingTimer.start();
        
        this->staticDataInitialized = 1;
    }
    
    this->MFNHashTypePlainMutex.unlock();

    // Retrieve our client ID.
    this->ClientId = this->Workunit->GetClientId();
    sprintf(statusBuffer, "Td %d: CID %d.", this->threadId, this->ClientId);
    this->Display->addStatusLine(statusBuffer);

    // If the device ID is 0, the device is the fastest in the system.
    // This is true for CUDA, and possibly OpenCL.  If this thread is for
    // a non-zero device, wait a second.  This allows the fastest GPU to
    // take the work.  Otherwise, we don't care what order they enter.
    if (this->gpuDeviceId) {
        CHSleep(1);
    }
    // Same for CPU devices.  They should NOT be the first device used!
    if (this->numberThreads) {
        CHSleep(2);
    }

    // Do all the device-specific setup.
    this->setupDevice();

    // Reset the per-step data for the new password length.
    this->perStep = 0;

    // Allocate the thread and GPU memory.
    this->allocateThreadAndDeviceMemory();
    
    // Build the OpenCL binaries if needed.
    this->doKernelSetup();

    // Copy all the run data to the device.
    this->copyDataToDevice();

    // Copy the kernel-specific constant data to the device.
    this->copyConstantDataToDevice();

    // I... *think* we're ready to rock!
    // As long as we aren't supposed to exit, keep running.
    while(!global_interface.exit && !this->threadRendezvous) {
        this->MFNHashTypePlainMutex.lock();
        WU = this->Workunit->GetNextWorkunit(ClientId);
        this->MFNHashTypePlainMutex.unlock();
        
        // If we are told to wait, wait around for a while.  A terminate
        // flag will break from the loop.
        while (WU.Flags == WORKUNIT_DELAY && !this->threadRendezvous && !global_interface.exit) {
            network_printf("Workunit wait requested - thread %d waiting.\n",
                    this->threadId);
            // If sleeping, speed is 0.
            this->Display->setThreadCrackSpeed(this->threadId, 0);
            CHSleep(1);
            this->MFNHashTypePlainMutex.lock();
            WU = this->Workunit->GetNextWorkunit(ClientId);
            this->MFNHashTypePlainMutex.unlock();
        }
        
        if (!WU.IsValid || (WU.Flags == WORKUNIT_TERMINATE)) {
            // If a null workunit came in, rendezvous the threads.
            this->threadRendezvous = 1;
            // Workunit came back null -
            sprintf(statusBuffer, "Td %d: out of WU.", this->threadId);
            this->Display->addStatusLine(statusBuffer);
            break;
        }
        if (this->CommandLineData->GetDevDebug()) {
            printf("Thread %d has workunit ID %d\n", this->threadId, WU.WorkUnitID);
        }
        // Do any per-workunit setup needed.
        //this->doPerWorkunitDeviceSetup();
        if (this->hashAttributes.hashUsesSalt) {
            this->MFNHashTypeSaltedMutex.lock();
            this->setupActiveSaltArrays();
            this->copySaltArraysToDevice();
            this->numberUniqueSalts = this->activeSalts.size();
            this->MFNHashTypeSaltedMutex.unlock();
        }
        
        if (WU.NumberWordsLoaded) {
            this->RunGPUWorkunitWL(&WU);
        } else {
            this->RunGPUWorkunitBF(&WU);
        }

        // If we are NOT aborting, submit the unit.
        // If we are force-exiting, do not submit the workunit!
        if (!global_interface.exit) {
            this->Workunit->SubmitWorkunit(WU);
        }
        this->Display->Refresh();
        //sprintf(this->statusBuffer, "WU rate: %0.1f", this->Workunit->GetAverageRate());
        //this->Display->addStatusLine(this->statusBuffer);
    }

    // Done with cracking - out of workunits.  Clean up & wait.

    // Free memory.
    this->freeThreadAndDeviceMemory();
    // Do final device teardown.
    this->teardownDevice();
    // Report speed of 0.
    this->Display->setThreadCrackSpeed(this->threadId, 0);


    // Wait until all workunits are back from remote systems.
    if (this->Workunit->GetNumberOfCompletedWorkunits() < this->Workunit->GetNumberOfWorkunits()) {
        sprintf(statusBuffer, "Waiting for workunits...");
        this->Display->addStatusLine(statusBuffer);
    }

    while (this->Workunit->GetNumberOfCompletedWorkunits() < this->Workunit->GetNumberOfWorkunits()) {
        CHSleep(1);
        //printf("Completed WU: %d\n", this->Workunit->GetNumberOfCompletedWorkunits());
        //printf("Total WU: %d\n", this->Workunit->GetNumberOfWorkunits());
        this->Display->Refresh();
        // Make termination work properly for the server
        if (global_interface.exit) {
            break;
        }
    }
    this->staticDataInitialized = 0;
}


// This is the GPU thread where we do the per-GPU tasks.
void MFNHashTypePlain::GPU_Thread() {
    trace_printf("MFNHashTypePlain::GPU_Thread()\n");
}

void MFNHashTypePlain::RunGPUWorkunitBF(MFNWorkunitRobustElement* WU) {
    trace_printf("MFNHashTypePlain::RunGPUWorkunitBF()\n");

    /**
     * High-res timer - this should work properly on both Windows & Posix.
     */
    CHHiresTimer Timer, WorkunitTimer;

    uint64_t perThread, start_point = 0;
    uint64_t step_count = 0;
    uint64_t tempPerStep = 0;
    
    /**
     * If iterations are being used, iterationsPerPass is set to the total
     * number of iterations.  The iterationIndexes is populated with the total
     * iteration sum for each salt, such that the salt can be calculated from
     * the iteration count.  This... is a trial, and it may change as I find out
     * what does and doesn't work.
     */
    uint64_t iterationsPerPass = 1;
    float averageIterations = 1.0;
    std::vector<uint64_t> iterationIndexes;

    WorkunitTimer.start();
    
    // Kernel run time: seconds
    float ref_time = 0.0f;
    // Kernel run time: Milliseconds
    float ref_time_ms = 0.0f;
    
    float ref_time_total = 0.0f;

    // Default number per steps.  As this is updated quickly, this just needs to be in the ballpark.
    if (this->perStep == 0) {
        this->perStep = 50;
    }

    klaunch_printf("Thread %d total kernel width: %d\n", this->threadId, this->TotalKernelWidth);
    klaunch_printf("Thread %d blocks/threads/vec: %d/%d/%d\n", this->threadId, this->GPUBlocks, this->GPUThreads, this->VectorWidth);
    
    // Calculate how many iterations per thread - divide total by the number of
    // total threads, then add one to deal with truncation.
    perThread = WU->EndPoint - WU->StartPoint + 1;
    perThread /= (this->TotalKernelWidth);
    perThread++;
    
    klaunch_printf("Total kernel width: %d\n", this->TotalKernelWidth);
    klaunch_printf("perThread: %d\n", perThread);

    if (!this->hashAttributes.hashUsesSalt) {
        // Round perThread up to a multiple of charset length.
        // Reduces divergent branches in vector kernels.
        while (perThread % this->charsetLengths[0]) {
            perThread++;
        }
        klaunch_printf("divergence adjusted perThread to: %d\n", perThread);
    }

    // Set up the password start points for loading as blocks.
    this->setStartPasswords32(perThread, start_point + WU->StartPoint);
    // Copy them to the GPU.
    this->copyStartPointsToDevice();

    /**
     * For salted hashes, after creating the start passwords, multiply by the
     * number of unique salts, so each "step" is a pass/salt pair.  This is done
     * by the iteration count algorithm automatically, so it is only done if the
     * hash is NOT iterated.
     */
    if (this->hashAttributes.hashUsesSalt &&
            !this->hashAttributes.hashUsesIterationCount) {
        klaunch_printf("Number unique salts: %lu\n", this->numberUniqueSalts);
        perThread *= this->numberUniqueSalts;
        klaunch_printf("perThread: %d\n", perThread);
    }
    
    // If the hash is using iteration counts, do some precalculations.
    if (this->hashAttributes.hashUsesIterationCount) {
        iterationsPerPass = 0;
        // Push back the first salt start point.
        iterationIndexes.push_back(0);
        for (size_t i = 0; i < this->activeIterationCounts.size(); i++) {
            iterationsPerPass += this->activeIterationCounts[i];
            iterationIndexes.push_back(iterationsPerPass);
            klaunch_printf("Pushed %lu\n", iterationsPerPass);
        }
        
        perThread *= iterationsPerPass;
        klaunch_printf("Total iterations per pass: %lu\n", iterationsPerPass);
        klaunch_printf("New total perThread: %lu\n", perThread);
        // Set the average for rate calculations.
        averageIterations = (float)iterationsPerPass /
                (float)this->activeIterationCounts.size();
    }

    // Start the timer.
    Timer.start();

    while (start_point <= perThread) {
        step_count++;

        if ((start_point + this->perStep) > perThread) {
            klaunch_printf("start_point: %lu\n", start_point);
            klaunch_printf("per_thread: %lu\n", perThread);
            klaunch_printf("Will overrun by %lu\n", (start_point + this->perStep) - perThread);
            tempPerStep = this->perStep;
            this->perStep = (perThread - start_point) + 1;
            klaunch_printf("Final per_step: %lu\n", this->perStep);
        }
        
        // We sync here and wait for the GPU to finish.
        this->synchronizeThreads();

        ref_time = Timer.getElapsedTime();
        ref_time_ms = Timer.getElapsedTimeInMilliSec();
        klaunch_printf("ref_time: %f s\n", ref_time);
        ref_time_total += ref_time;
        
        // Run this roughly every second, or every step if target_ms is >500
        if ((step_count < 5) || (this->kernelTimeMs > 500) || (step_count % (1000 / this->kernelTimeMs) == 0)) {

            this->copyDeviceFoundPasswordsToHost();
            this->outputFoundHashes();
            if (this->CommandLineData->GetDebug()) {
                printf("ref_time: %f\n", ref_time_ms);
                printf("%0.2f%% done with WU\n", 100.0 * (float)start_point / (float)perThread);
            }

            // Only set the crack speed if we have set one and aren't on the
            // last step - this leads to glitches in speed.
            if ((step_count > 5) && !tempPerStep) {
                this->Display->setThreadCrackSpeed(this->threadId,
                        (float) ((float)this->TotalKernelWidth *
                        (float)this->perStep /
                        ((float)this->numberUniqueSalts * averageIterations)) 
                        / (ref_time));
                
                // If the time since the last network update has elapsed, send
                // the total cracking speed over the network.
                if (this->CommandLineData->GetIsNetworkClient()) {
                    if (this->NetworkSpeedReportingTimer.getElapsedTime() > 
                            MFN_NETWORK_REPORTING_INTERVAL) {
                        // Report the total speed & reset the timer.
                        this->NetworkSpeedReportingTimer.start();
                        MultiforcerGlobalClassFactory.getNetworkClientClass()->
                                submitSystemCrackingRate((uint64_t)this->
                                Display->getCurrentCrackRate());
                    }
                }
                
            }
            // If the current execution time is not correct, adjust.
            // Do not adjust this if it is the final step.
            if ((!tempPerStep) && (ref_time_ms > 0) && (step_count > 2) &&
                    ((ref_time_ms < (this->kernelTimeMs * 0.9)) ||
                    (ref_time_ms > (this->kernelTimeMs * 1.1)))) {
                this->perStep = (uint64_t) ((float) this->perStep *
                        ((float) this->kernelTimeMs / ref_time_ms));
                // Never set this to 0.
                if (this->perStep == 0) {
                    this->perStep = 1;
                }
                if (this->CommandLineData->GetDebug()) {
                    printf("\nThread %d Adjusting passwords per step to %d\n",
                            this->gpuDeviceId, (unsigned int) this->perStep);
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


        //this->copyStartPointsToDevice();
        //this->synchronizeThreads();
        Timer.start();

        if (this->hashAttributes.hashUsesIterationCount) {
            // Calculate which salt and which iteration we are starting on.
            
            // Determine how far through a "pass" (all salts, all iterations) we
            // are.
            uint64_t pass_progress = start_point % iterationsPerPass;
            klaunch_printf("start_point in iteration calc: %lu\n", start_point);
            klaunch_printf("pass_progress: %lu\n", pass_progress);
            
            // Determine which salt we are on.
            size_t low = 0;
            size_t high = iterationIndexes.size();
            size_t index = high / 2;
            
            // Binary search through the tree to find the right point.
            // TODO: BINARY SEARCH
            for (size_t i = 1; i < iterationIndexes.size(); i++) {
                if ((pass_progress >= iterationIndexes[i - 1]) && 
                    (pass_progress < iterationIndexes[i])) {
                    this->saltStartOffset = i - 1;
                    break;
                }
            }
            
            klaunch_printf("Found salt at position %lu\n", this->saltStartOffset);
            this->iterationStartOffset = pass_progress - iterationIndexes[this->saltStartOffset];
            klaunch_printf("Setting start iteration %lu\n", this->iterationStartOffset);
            
        } else {
            this->saltStartOffset = start_point % this->numberUniqueSalts;
            this->iterationStartOffset = 0;
        }
        
        klaunch_printf("Launching kernel: \n");
        klaunch_printf("  start_point: %lu\n", start_point);
        klaunch_printf("  perStep: %lu\n", this->perStep);
        klaunch_printf("  num salts: %lu\n", this->numberUniqueSalts);
        klaunch_printf("  saltStartOffset: %lu\n", this->saltStartOffset);

        this->launchKernel();

        // Increment start point by however many we did
        start_point += this->perStep;
    }

    this->synchronizeThreads();
    
    // Perform a final rate calculation.
    // In some cases, the device is too fast for the normal speed reporting
    // to get triggered.
    Timer.stop();
    ref_time = Timer.getElapsedTime();
    this->Display->setThreadCrackSpeed(this->threadId,
        (float) ((float)this->TotalKernelWidth *
        (float)this->perStep /
        ((float)this->numberUniqueSalts * averageIterations))
        / (ref_time));

    this->copyDeviceFoundPasswordsToHost();
    this->outputFoundHashes();
    
    WorkunitTimer.stop();
    
    klaunch_printf("Workunit rate: %f\n", (WU->EndPoint - WU->StartPoint) / WorkunitTimer.getElapsedTime());
    klaunch_printf("Workunit timer: %f\n", WorkunitTimer.getElapsedTime());
    klaunch_printf("ref_time_total: %f\n", ref_time_total);
    
    if (tempPerStep) {
        klaunch_printf("Correcting perStep from current %lu to perm %lu\n", this->perStep, tempPerStep);
        this->perStep = tempPerStep;
        tempPerStep = 0;
    }
    
    return;
}


void MFNHashTypePlain::RunGPUWorkunitWL(MFNWorkunitRobustElement* WU) {
    trace_printf("MFNHashTypePlain::RunGPUWorkunitWL()\n");

    /**
     * High-res timer - this should work properly on both Windows & Posix.
     */
    CHHiresTimer Timer, WorkunitTimer;

    klaunch_printf("Copying words to device...\n");
    klaunch_printf("Block length: %d\n", WU->WordBlockLength);
    klaunch_printf("Number words: %d\n", WU->WordLengths.size());
    this->wordlistBlockLength = WU->WordBlockLength;
    
    // Convert wordlists
    std::vector<uint8_t> convertedWordlistLengths;
    std::vector<uint32_t> convertedWordlistBlocks;
    
    this->covertWordlist32(
        WU->WordLengths,
        WU->WordlistData,
        convertedWordlistLengths,
        convertedWordlistBlocks,
        (this->VectorWidth * 4));

    this->copyWordlistToDevice(
        convertedWordlistLengths,
        convertedWordlistBlocks);
    
    uint64_t perThread, start_point = 0;
    uint64_t step_count = 0;
    uint64_t tempPerStep = 0;
    // If the kernel is not "full" this will allow speed adjustment without
    // blowing out future kernels on execution time with a full kernel.
    uint64_t notFullStorePerStep = 0;
    // Store the effective width - wordlist size if less than actual width.
    uint32_t effectiveKernelWidth = this->TotalKernelWidth;
    
    if (WU->WordLengths.size() < effectiveKernelWidth) {
        effectiveKernelWidth = WU->WordLengths.size();
    }

    /**
     * If iterations are being used, iterationsPerPass is set to the total
     * number of iterations.  The iterationIndexes is populated with the total
     * iteration sum for each salt, such that the salt can be calculated from
     * the iteration count.  This... is a trial, and it may change as I find out
     * what does and doesn't work.
     */
    uint64_t iterationsPerPass = 1;
    float averageIterations = 1.0;
    std::vector<uint64_t> iterationIndexes;

    WorkunitTimer.start();
    
    // Kernel run time: seconds
    float ref_time = 0.0f;
    // Kernel run time: Milliseconds
    float ref_time_ms = 0.0f;
    
    float ref_time_total = 0.0f;

    // Default number per steps.  As this is updated quickly, this just needs to be in the ballpark.
    if (this->perStep == 0) {
        this->perStep = 500;
    }
    
    // If the kernel is not full, store the previous perStep for reset later.
    if ((WU->WordLengths.size() < (this->TotalKernelWidth))) {
        notFullStorePerStep = this->perStep;
    }

    klaunch_printf("Thread %d total kernel width: %d\n", this->threadId, this->TotalKernelWidth);
    klaunch_printf("Thread %d blocks/threads/vec: %d/%d/%d\n", this->threadId, this->GPUBlocks, this->GPUThreads, this->VectorWidth);
    
    // Calculate how many iterations per thread - divide total by the number of
    // total threads, then add one to deal with truncation.
    perThread = convertedWordlistLengths.size() + 1;
    perThread /= (this->TotalKernelWidth);
    perThread++;
    
    klaunch_printf("Total kernel width: %d\n", this->TotalKernelWidth);
    klaunch_printf("perThread: %d\n", perThread);

    /**
     * For salted hashes, after creating the start passwords, multiply by the
     * number of unique salts, so each "step" is a pass/salt pair.  This is done
     * by the iteration count algorithm automatically, so it is only done if the
     * hash is NOT iterated.
     */
    if (this->hashAttributes.hashUsesSalt &&
            !this->hashAttributes.hashUsesIterationCount) {
        klaunch_printf("Number unique salts: %lu\n", this->numberUniqueSalts);
        perThread *= this->numberUniqueSalts;
        klaunch_printf("perThread: %d\n", perThread);
    }
    
    // If the hash is using iteration counts, do some precalculations.
    if (this->hashAttributes.hashUsesIterationCount) {
        iterationsPerPass = 0;
        // Push back the first salt start point.
        iterationIndexes.push_back(0);
        for (size_t i = 0; i < this->activeIterationCounts.size(); i++) {
            iterationsPerPass += this->activeIterationCounts[i];
            iterationIndexes.push_back(iterationsPerPass);
            klaunch_printf("Pushed %lu\n", iterationsPerPass);
        }
        
        perThread *= iterationsPerPass;
        klaunch_printf("Total iterations per pass: %lu\n", iterationsPerPass);
        klaunch_printf("New total perThread: %lu\n", perThread);
        // Set the average for rate calculations.
        averageIterations = (float)iterationsPerPass /
                (float)this->activeIterationCounts.size();
    }

    
    // Start the timer.
    Timer.start();

    while (start_point <= perThread) {
        step_count++;

        // We sync here and wait for the GPU to finish.
        this->synchronizeThreads();

        ref_time = Timer.getElapsedTime();
        ref_time_ms = Timer.getElapsedTimeInMilliSec();
        klaunch_printf("ref_time: %f s\n", ref_time);
        ref_time_total += ref_time;
        
        // Run this roughly every second, or every step if target_ms is >500
        if ((step_count < 5) || (this->kernelTimeMs > 500) || (step_count % (1000 / this->kernelTimeMs) == 0)) {

            this->copyDeviceFoundPasswordsToHost();
            this->outputFoundHashes();
            if (this->CommandLineData->GetDebug()) {
                printf("ref_time: %f\n", ref_time_ms);
                printf("%0.2f%% done with WU\n", 100.0 * (float)start_point / (float)perThread);
            }

            // Only set the crack speed if we have set one and aren't on the
            // last step - this leads to glitches in speed.
            if ((step_count > 5) && !tempPerStep) {
                this->Display->setThreadCrackSpeed(this->threadId,
                        (float) ((float)effectiveKernelWidth *
                        (float)this->perStep /
                        ((float)this->numberUniqueSalts * averageIterations)) 
                        / (ref_time));
                
                // If the time since the last network update has elapsed, send
                // the total cracking speed over the network.
                if (this->CommandLineData->GetIsNetworkClient()) {
                    if (this->NetworkSpeedReportingTimer.getElapsedTime() > 
                            MFN_NETWORK_REPORTING_INTERVAL) {
                        // Report the total speed & reset the timer.
                        this->NetworkSpeedReportingTimer.start();
                        MultiforcerGlobalClassFactory.getNetworkClientClass()->
                                submitSystemCrackingRate((uint64_t)this->
                                Display->getCurrentCrackRate());
                    }
                }
                
            }
            // If the current execution time is not correct, adjust.
            // Only adjust if there's enough work to fill the kernel.
            if (/*(WU->WordLengths.size() >= (this->TotalKernelWidth)) &&*/
                    (ref_time_ms > 0) && (step_count > 2) &&
                    ((ref_time_ms < (this->kernelTimeMs * 0.9)) ||
                    (ref_time_ms > (this->kernelTimeMs * 1.1)))) {
                this->perStep = (uint64_t) ((float) this->perStep *
                        ((float) this->kernelTimeMs / ref_time_ms));
                // Never set this to 0.
                if (this->perStep == 0) {
                    this->perStep = 1;
                }
                if (this->CommandLineData->GetDebug()) {
                    printf("\nThread %d Adjusting passwords per step to %d\n",
                            this->gpuDeviceId, (unsigned int) this->perStep);
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


        //this->copyStartPointsToDevice();
        //this->synchronizeThreads();
        Timer.start();
        if (this->hashAttributes.hashUsesIterationCount) {
            // Calculate which salt and which iteration we are starting on.
            
            // Determine how far through a "pass" (all salts, all iterations) we
            // are.
            uint64_t pass_progress = start_point % iterationsPerPass;
            klaunch_printf("start_point in iteration calc: %lu\n", start_point);
            klaunch_printf("pass_progress: %lu\n", pass_progress);
            
            // Determine which salt we are on.
            size_t low = 0;
            size_t high = iterationIndexes.size();
            size_t index = high / 2;
            
            // Binary search through the tree to find the right point.
            // TODO: BINARY SEARCH
            for (size_t i = 1; i < iterationIndexes.size(); i++) {
                if ((pass_progress >= iterationIndexes[i - 1]) && 
                    (pass_progress < iterationIndexes[i])) {
                    this->saltStartOffset = i - 1;
                    break;
                }
            }
            
            klaunch_printf("Found salt at position %lu\n", this->saltStartOffset);
            this->iterationStartOffset = pass_progress - iterationIndexes[this->saltStartOffset];
            klaunch_printf("Setting start iteration %lu\n", this->iterationStartOffset);
            this->startStep = start_point / iterationsPerPass;
            klaunch_printf("Setting start step %lu\n", this->startStep);
        } else {
            this->saltStartOffset = start_point % this->numberUniqueSalts;
            this->startStep = start_point / this->numberUniqueSalts;
            this->iterationStartOffset = 0;
        }

        
        // This should go down here, instead of before the adjustment point!
        if ((start_point + this->perStep) > perThread) {
            klaunch_printf("start_point: %lu\n", start_point);
            klaunch_printf("per_thread: %lu\n", perThread);
            klaunch_printf("Will overrun by %lu\n", (start_point + this->perStep) - perThread);
            tempPerStep = this->perStep;
            this->perStep = (perThread - start_point) + 1;
            klaunch_printf("Final per_step: %lu\n", this->perStep);
        }

        
        klaunch_printf("Launching kernel: \n");
        klaunch_printf("  start_point: %lu\n", start_point);
        klaunch_printf("  perStep: %lu\n", this->perStep);
        klaunch_printf("  num salts: %lu\n", this->numberUniqueSalts);
        klaunch_printf("  saltStartOffset: %lu\n", this->saltStartOffset);
        klaunch_printf("  startStep: %lu\n", this->startStep);

        this->launchKernel();

        // Increment start point by however many we did
        start_point += this->perStep;
    }

    this->synchronizeThreads();
    
    // Perform a final rate calculation.
    // In some cases, the device is too fast for the normal speed reporting
    // to get triggered.
    Timer.stop();
    ref_time = Timer.getElapsedTime();
    this->Display->setThreadCrackSpeed(this->threadId,
        (float) ((float)effectiveKernelWidth *
        (float)this->perStep /
        ((float)this->numberUniqueSalts * averageIterations))
        / (ref_time));

    this->copyDeviceFoundPasswordsToHost();
    this->outputFoundHashes();
    
    WorkunitTimer.stop();
    
    klaunch_printf("Workunit rate: %f\n", (WU->EndPoint - WU->StartPoint) / WorkunitTimer.getElapsedTime());
    klaunch_printf("Workunit timer: %f\n", WorkunitTimer.getElapsedTime());
    klaunch_printf("ref_time_total: %f\n", ref_time_total);
    
    if (tempPerStep) {
        klaunch_printf("Correcting perStep from current %lu to perm %lu\n", this->perStep, tempPerStep);
        this->perStep = tempPerStep;
        tempPerStep = 0;
    }
    
    // Restore the per-step if the kernel was not full.
    if (notFullStorePerStep) {
        klaunch_printf("Correcting perStep with not full data\n");
        this->perStep = notFullStorePerStep;
        notFullStorePerStep = 0;
    }
    
    return;
}

void MFNHashTypePlain::createLookupBitmaps() {
    trace_printf("MFNHashTypePlain::createLookupBitmaps()\n");
    
    // This involves creating bitmaps based on the provided hashes.
    // If the hash is big endian, they will be reversed compared to the hash.
    
    // Create bitmaps a (8kb and 128mb)
    if (this->hashLengthBytes >= 4) {
        static_printf("Creating bitmaps for word 0/a\n");
        this->createArbitraryBitmap(0, this->activeHashesProcessed, this->sharedBitmap8kb_a, (8 * 1024));
        this->createArbitraryBitmap(0, this->activeHashesProcessed, this->globalBitmap128mb_a, (128 * 1024 * 1024));
        this->createArbitraryBitmap(0, this->activeHashesProcessed, this->globalBitmap256kb_a, (256 * 1024));
        this->createArbitraryBitmap(0, this->activeHashesProcessed, this->sharedBitmap16kb_a, (16 * 1024));
        this->createArbitraryBitmap(0, this->activeHashesProcessed, this->sharedBitmap32kb_a, (32 * 1024));
    }
    if (this->hashLengthBytes >= 8) {
        static_printf("Creating bitmaps for word 1/b\n");
        this->createArbitraryBitmap(1, this->activeHashesProcessed, this->sharedBitmap8kb_b, (8 * 1024));
        this->createArbitraryBitmap(1, this->activeHashesProcessed, this->globalBitmap128mb_b, (128 * 1024 * 1024));
        this->createArbitraryBitmap(1, this->activeHashesProcessed, this->globalBitmap256kb_b, (256 * 1024));
        this->createArbitraryBitmap(1, this->activeHashesProcessed, this->sharedBitmap16kb_b, (16 * 1024));
        this->createArbitraryBitmap(1, this->activeHashesProcessed, this->sharedBitmap32kb_b, (32 * 1024));
    }
    if (this->hashLengthBytes >= 12) {
        static_printf("Creating bitmaps for word 2/c\n");
        this->createArbitraryBitmap(2, this->activeHashesProcessed, this->sharedBitmap8kb_c, (8 * 1024));
        this->createArbitraryBitmap(2, this->activeHashesProcessed, this->globalBitmap128mb_c, (128 * 1024 * 1024));
        this->createArbitraryBitmap(2, this->activeHashesProcessed, this->globalBitmap256kb_c, (256 * 1024));
        this->createArbitraryBitmap(2, this->activeHashesProcessed, this->sharedBitmap16kb_c, (16 * 1024));
        this->createArbitraryBitmap(2, this->activeHashesProcessed, this->sharedBitmap32kb_c, (32 * 1024));
    }
    if (this->hashLengthBytes >= 16) {
        static_printf("Creating bitmaps for word 3/d\n");
        this->createArbitraryBitmap(3, this->activeHashesProcessed, this->sharedBitmap8kb_d, (8 * 1024));
        this->createArbitraryBitmap(3, this->activeHashesProcessed, this->globalBitmap128mb_d, (128 * 1024 * 1024));
        this->createArbitraryBitmap(3, this->activeHashesProcessed, this->globalBitmap256kb_d, (256 * 1024));
        this->createArbitraryBitmap(3, this->activeHashesProcessed, this->sharedBitmap16kb_d, (16 * 1024));
        this->createArbitraryBitmap(3, this->activeHashesProcessed, this->sharedBitmap32kb_d, (32 * 1024));
    }
    
}

void MFNHashTypePlain::createArbitraryBitmap(uint8_t startWord,
        std::vector<std::vector<uint8_t> > &hashList, std::vector<uint8_t> &bitmap,
        uint32_t bitmapSizeBytes) {

    uint32_t bitmapIndex;
    uint8_t  bitmapByte;
    uint64_t passwordIndex;
    uint32_t bitmapMask;

    if (!isPowerOfTwo(bitmapSizeBytes)) {
        printf("Error!  Bitmap size not a power of 2!\n");
        exit(1);
    }

    // Set the bitmap mask - size - 1 for and masking.
    bitmapMask = (bitmapSizeBytes - 1);

    // Step 1: Set the vector to whatever is specified.
    bitmap.resize(bitmapSizeBytes);
    // Step 2: Clear the vector
    memset(&bitmap[0], 0, bitmapSizeBytes);


    for (passwordIndex = 0; passwordIndex < hashList.size(); passwordIndex++) {
        if (0 /*this->HashIsBigEndian*/) {
            // Big endian hash - read in as a big endian value
            bitmapIndex =
                    ((uint32_t)hashList.at(passwordIndex).at((startWord * 4) + 0) << 24) +
                    ((uint32_t)hashList.at(passwordIndex).at((startWord * 4) + 1) << 16) +
                    ((uint32_t)hashList.at(passwordIndex).at((startWord * 4) + 2) <<  8) +
                    ((uint32_t)hashList.at(passwordIndex).at((startWord * 4) + 3) <<  0);
        } else {
            // Little endian hash - take bytes 0 & 1 in the word as the low value (swapped).
            bitmapIndex =
                    ((uint32_t)hashList.at(passwordIndex).at((startWord * 4) + 0) <<  0) +
                    ((uint32_t)hashList.at(passwordIndex).at((startWord * 4) + 1) <<  8) +
                    ((uint32_t)hashList.at(passwordIndex).at((startWord * 4) + 2) << 16) +
                    ((uint32_t)hashList.at(passwordIndex).at((startWord * 4) + 3) << 24);
        }
        //printf("bitmapIndex: %08x\n", bitmapIndex);

        // Set the byte by shifting left by the lower 3 bits in the index
        bitmapByte = 0x01 << (bitmapIndex & 0x0007);
        // Determine the byte offset by shifting right 3 bits.
        bitmapIndex = bitmapIndex >> 3;
        // Mask off the lower 27 bits
        bitmapIndex &= bitmapMask;

        //printf("bitmapByte: %02x\n", bitmapByte);
        //printf("bitmapIndex: %08x\n", bitmapIndex);

        if (bitmapIndex >= bitmapSizeBytes) {
            printf("FATAL ERROR: Bitmap index beyond bound of bitmap!\n");
            exit(1);
        }

        // Add the bit into the bitmap
        bitmap[bitmapIndex] |= bitmapByte;
    }
}


void MFNHashTypePlain::sortHashes() {
    trace_printf("MFNHashTypePlain::sortHashes()\n");
    if (0 /*this->HashIsBigEndian*/) {
        // Big endian sort
        // Sort hashes with the big endian sort predicate.  This will interpret
        // them as 00112233 < 11111111
        std::sort(this->activeHashesProcessed.begin(),
                this->activeHashesProcessed.end(), hashBigEndianSortPredicate);
    } else {
        // Little endian sort
        // Sort hashes with the little endian sort predicate.  This will interpret
        // them as 00112233 > 11111111
        std::sort(this->activeHashesProcessed.begin(),
                this->activeHashesProcessed.end(), hashLittleEndianSortPredicate);
   }
    this->activeHashesProcessed.erase(
        std::unique(this->activeHashesProcessed.begin(), this->activeHashesProcessed.end(), hashUniquePredicate),
        this->activeHashesProcessed.end());

    // Debug print hashes
    /*
    printf("sortHashes() printout\n");
    uint32_t i, j;
    for (i = 0; i < this->activeHashesProcessed.size(); i++) {
        for (j = 0; j < this->activeHashesProcessed[i].size(); j++) {
            printf("%02x", this->activeHashesProcessed[i][j]);
        }
        printf("\n");
    }
    */
    
}


void MFNHashTypePlain::copyHashesIntoDeviceFormat() {
    trace_printf("MFNHashTypePlain::copyHashesIntoDeviceFormat()\n");

    uint64_t hashIndex;

    /** Convert the processed hashlist into a single vector suited to copying
     * to the GPUs.  This will be the same for CUDA and OpenCL (and can be
     * used by the CPU as well.
     */

    // Reserve the right amount of space in the main vector - number of elements * hashlength
    this->activeHashesProcessedDeviceformat.resize(
            this->activeHashesProcessed.size() * this->hashLengthBytes);

    for (hashIndex = 0; hashIndex < this->activeHashesProcessed.size(); hashIndex++) {
        memcpy(&this->activeHashesProcessedDeviceformat[hashIndex * this->hashLengthBytes],
                &this->activeHashesProcessed[hashIndex][0], this->hashLengthBytes);
    }

    static_printf("Created common hash array of %d bytes.\n", this->activeHashesProcessedDeviceformat.size());

    if (0) {
        for (hashIndex = 0; hashIndex < this->activeHashesProcessed.size(); hashIndex++) {
            for (int j = 0; j < this->hashLengthBytes; j++) {
                printf("%02x", this->activeHashesProcessedDeviceformat[hashIndex * this->hashLengthBytes + j]);
            }
            printf("\n");
        }
    }
    
}

void MFNHashTypePlain::setupCharsetArrays() {
    trace_printf("MFNHashTypePlain::setupCharsetArrays()\n");

    uint32_t charsetItemsToCopy, i, j;


    // Ensure that we zero unused elements.
    this->charsetLengths.resize(this->passwordLength, 0);

    // Step 1: Set up the charset array - CHARSET_LENGTH elements per length.
    if (this->currentCharset.size() == 1) {
        charsetItemsToCopy = 1;

        // If the charset is single (length 1), then only allocate CHARSET_LENGTH bytes for it.
        this->charsetForwardLookup.resize(MFN_HASH_TYPE_PLAIN_MAX_CHARSET_LENGTH, 0);
        this->charsetReverseLookup.resize(MFN_HASH_TYPE_PLAIN_MAX_CHARSET_LENGTH, 0);

        this->charsetLengths[0] = this->currentCharset[0].size();
        
        for (i = 0; i < this->currentCharset[0].size(); i++) {
            this->charsetForwardLookup[i] = this->currentCharset[0][i];
            this->charsetReverseLookup[this->currentCharset[0][i]] = i;
        }

    } else {
        // Vector is multiple - ensure it is long enough, then copy it.
        if (this->currentCharset.size() < this->passwordLength) {
            printf("Error!  Multiposition charset is shorter than password!\n");
            exit(1);
        }
        charsetItemsToCopy = this->passwordLength;

        // Make room!  PassLength * CHARSET_LENGTH
        this->charsetForwardLookup.resize(this->passwordLength * 
                MFN_HASH_TYPE_PLAIN_MAX_CHARSET_LENGTH, 0);
        this->charsetReverseLookup.resize(this->passwordLength * 
                MFN_HASH_TYPE_PLAIN_MAX_CHARSET_LENGTH, 0);

        for (i = 0; i < charsetItemsToCopy; i++) {
            this->charsetLengths[i] = this->currentCharset[i].size();
            for (j = 0; j < this->currentCharset[i].size(); j++) {
                this->charsetForwardLookup[(i * MFN_HASH_TYPE_PLAIN_MAX_CHARSET_LENGTH) + j] = 
                        this->currentCharset[i][j];
                this->charsetReverseLookup[(i * MFN_HASH_TYPE_PLAIN_MAX_CHARSET_LENGTH) + 
                        this->currentCharset[i][j]] = j;
            }
        }
    }
}

void MFNHashTypePlain::setStartPasswords32(uint64_t perThread, uint64_t startPoint) {
    trace_printf("MFNHashTypePlain::setStartPasswords32()\n");

    uint64_t threadId, threadStartPoint;
    uint32_t characterPosition;


    // Resize the vector to the needed number of bytes.  This will possibly
    // have waste space at the end, but will be loaded as words, so needs
    // to be a multiple of 4 length.  Init to 0, so the unused bytes are
    // null.
    this->HostStartPasswords32.resize(this->TotalKernelWidth * this->passwordLengthWords, 0);

    
    if (this->isSingleCharset) {
        klaunch_printf("Calculating start points for a single charset.\n");
        // Copy the current charset length into a local variable for speed.
        uint8_t currentCharsetLength = this->currentCharset.at(0).size();

        for (threadId = 0; threadId < this->TotalKernelWidth; threadId++) {

            threadStartPoint = threadId * perThread + startPoint;
            //printf("Thread %u, startpoint %lu, perThread %d\n", threadId, threadStartPoint, perThread);

            // Loop through all the character positions.  This is easier than a case statement.
            for (characterPosition = 0; characterPosition < this->passwordLength; characterPosition++) {
                // Base offset: b0 starts at (kernelWidth * 0), b1 starts at (kernelWidth * 4), etc.
                uint32_t baseOffset = ((characterPosition / 4) * this->TotalKernelWidth * 4);
                // Character offset: baseOffset + (threadId * 4) + (characterPos % 4)
                if (this->HashIsBigEndian) {
                    // Big endian - swap Word bytes
                    this->HostStartPasswords32[baseOffset + (threadId * 4) + (3 - (characterPosition % 4))] =
                            this->currentCharset[0][(uint8_t)(threadStartPoint % currentCharsetLength)];
                } else {
                    // Little endian - do as normal.
                    this->HostStartPasswords32[baseOffset + (threadId * 4) + (characterPosition % 4)] =
                            this->currentCharset[0][(uint8_t)(threadStartPoint % currentCharsetLength)];
                }
                threadStartPoint /= currentCharsetLength;
            }
            // Set the padding bit.
            if (this->HashIsBigEndian) {
                this->HostStartPasswords32[((this->passwordLength / 4) * this->TotalKernelWidth * 4)
                        + (threadId * 4) + (3 - (this->passwordLength % 4))] = 0x80;
            } else {
                this->HostStartPasswords32[((this->passwordLength / 4) * this->TotalKernelWidth * 4)
                        + (threadId * 4) + (this->passwordLength % 4)] = 0x80;
            }
        }
    } else {
        klaunch_printf("Calculating start points for a multiple charset.\n");
        // Copy the current charset length into a local variable for speed.
        uint8_t currentCharsetLength = this->currentCharset.at(0).size();

        for (threadId = 0; threadId < this->TotalKernelWidth; threadId++) {

            threadStartPoint = threadId * perThread + startPoint;
            //printf("Thread %u, startpoint %lu, perThread %d\n", threadId, threadStartPoint, perThread);

            // Loop through all the character positions.  This is easier than a case statement.
            for (characterPosition = 0; characterPosition < this->passwordLength; characterPosition++) {
                // Base offset: b0 starts at (kernelWidth * 0), b1 starts at (kernelWidth * 4), etc.
                uint32_t baseOffset = ((characterPosition / 4) * this->TotalKernelWidth * 4);
                // Character offset: baseOffset + (threadId * 4) + (characterPos % 4)
                if (this->HashIsBigEndian) {
                    this->HostStartPasswords32[baseOffset + (threadId * 4) + (3 - (characterPosition % 4))] =
                            this->currentCharset[characterPosition][(uint8_t)(threadStartPoint % currentCharsetLength)];
                } else {
                    this->HostStartPasswords32[baseOffset + (threadId * 4) + (characterPosition % 4)] =
                            this->currentCharset[characterPosition][(uint8_t)(threadStartPoint % currentCharsetLength)];
                }
                threadStartPoint /= currentCharsetLength;
            }
            // Set the padding bit.
            if (this->HashIsBigEndian) {
                this->HostStartPasswords32[((this->passwordLength / 4) * this->TotalKernelWidth * 4)
                        + (threadId * 4) + (3 - (this->passwordLength % 4))] = 0x80;
            } else {
                this->HostStartPasswords32[((this->passwordLength / 4) * this->TotalKernelWidth * 4)
                        + (threadId * 4) + (this->passwordLength % 4)] = 0x80;
            }
        }
    }
}


void MFNHashTypePlain::convertData32(
        const std::vector<std::vector<uint8_t> > &inputData,
        char isBigEndian,
        char addPaddingBit,
        uint8_t dataOffset,
        std::vector<uint32_t> &dataLengths, 
        std::vector<uint32_t> &dataDeviceFormat
        ) {
    trace_printf("MFNHashTypePlain::convertData32()\n");
    uint32_t maximumDataLength = 0; // Max data length (with padding and offset)
    uint32_t dataWordsNeeded = 0; // Number of 32-bit words to fit all data
    uint32_t dataWord, dataCount = 0;
    std::vector<std::vector<uint8_t> >::const_iterator dataIt; // Iterator for data
    
    // Clear out any existing data.
    dataLengths.clear();
    dataLengths.reserve(inputData.size());
    dataDeviceFormat.clear();
    
    // Ensure dataOffset is in the range of 0-3
    dataOffset = dataOffset % 4;
    
    /**
     * Iterate through the salts and do several things: Get the length of each
     * salt and push it into the array, add the padding bit to the end if
     * requested, and determine the maximum salt length in use for creating the
     * device length array
     */
    for (dataIt = inputData.begin(); dataIt < inputData.end(); dataIt++) {
        // Set the length - this is the *actual* length of the data, without
        // any padding or offset factored in.
        dataLengths.push_back((uint32_t)dataIt->size());

        // Determine the maximum salt length.  We include the padding bit if
        // set - it will need space in the array!
        if (dataIt->size() > maximumDataLength) {
            maximumDataLength = dataIt->size();
        }
    }
    
    // If a padding bit is used, add one to the max data length.
    if (addPaddingBit) {
        maximumDataLength++;
    }
    // Add the offset to the max data size.
    maximumDataLength += dataOffset;
    //printf("Maximum salt length: %d\n", maximumDataLength);

    // Determine how many 32-bit words are needed to fit everything
    dataWordsNeeded = (maximumDataLength / 4);
    // If there are more bytes, add another word.
    if (maximumDataLength % 4) {
        dataWordsNeeded++;
    }
    //printf("Max salt words: %d\n", dataWordsNeeded);
    
    // Resize the device data vector.
    dataDeviceFormat.resize(inputData.size() * dataWordsNeeded);

    for (dataIt = inputData.begin(); dataIt < inputData.end(); dataIt++) {
        dataWord = 0;
        size_t dataSize = dataIt->size();
        uint8_t dataByte;
        
        // If a padding bit is being added, increase size to account for it.
        if (addPaddingBit) {
            dataSize++;
        }

        for (uint32_t i = 0; i < dataSize; i++) {
            // Data byte is either the value from the vector, or the padding bit
            // if needed.
            if (i < dataIt->size()) {
                dataByte = dataIt->at(i);
            } else {
                dataByte = 0x80;
            }
            
            // Insert the byte into the word in the proper spot.
            if (!isBigEndian) {
                // Hash is big endian.  Byte 0 is shifted 0.
                dataWord |= (uint32_t)dataByte << (((i + dataOffset) % 4) * 8);
            } else {
                // Hash is little endian.  Byte 0 gets shifted << 24
                dataWord |= (uint32_t)dataByte << (((3 - (i + dataOffset)) % 4) * 8);
            }
            
            // Check to see if the data word needs to be pushed back.  This is
            // true if a word is full (%3 == 0), or if it is the last word in
            // the salt (== ->size() - 1).  In any case, push the word and reset
            // the storage value.
            if ((((i + dataOffset) % 4) == 3) ||
                    (i == (dataSize - 1))) {
                if (((i + dataOffset) % 4) == 3) {
                    //printf("Pushing for %%4 - i: %d  doffset: %d\n", i, dataOffset);
                }
                if (i == (dataSize - 1)) {
                    //printf("Pushing for size\n");
                }
                // Offset in the salt array: (totalSalts * word + currentSalt)
                //printf("Shoved dataword 0x%08x into position %u\n", dataWord,
                //        inputData.size() * ((i + dataOffset) / 4) + dataCount);
                dataDeviceFormat[inputData.size() *
                        ((i + dataOffset) / 4) + dataCount] = dataWord;
                dataWord = 0;
            }
        }
        dataCount++;
    }
}

void MFNHashTypePlain::covertWordlist32(
        std::vector<uint8_t> &inputWordlistLengths,
        std::vector<uint32_t> &inputWordlistBlocks,
        std::vector<uint8_t> &outputWordlistLengths,
        std::vector<uint32_t> &outputWordlistBlocks,
        uint32_t byteAlignOffset) {
    
    uint32_t incomingDataCount;
    uint32_t outgoingDataCount;
    uint8_t blocksPerWord;

    // Sort out the alignment.
    
    incomingDataCount = inputWordlistLengths.size();
    outgoingDataCount = incomingDataCount;
    
    // Convert byte alignment to word alignment
    byteAlignOffset /= 4;
    
    //printf("Incoming data size: %d\n", incomingDataCount);
    // If things aren't aligned, add words until they are.
    if (outgoingDataCount % byteAlignOffset) {
        while (outgoingDataCount % byteAlignOffset) {
            outgoingDataCount++;
        }
    }
    //printf("Outgoing data size: %d\n", outgoingDataCount);
    
    // Determine how many blocks each word has.
    blocksPerWord = inputWordlistBlocks.size() / incomingDataCount;
    
    // Copy data to the output size and 0-pad.
    outputWordlistLengths.assign(inputWordlistLengths.begin(), inputWordlistLengths.end());
    outputWordlistLengths.resize(outgoingDataCount, 0);

    // Resize the output block list so it can be fully addressed.
    outputWordlistBlocks.resize(outgoingDataCount * blocksPerWord, 0);
    
    // Go about shuffling the data.
    for (uint32_t inputWord = 0; inputWord < incomingDataCount; inputWord++) {
        for (uint32_t inputBlock = 0; inputBlock < blocksPerWord; inputBlock++) {
            outputWordlistBlocks[(outgoingDataCount * inputBlock) + inputWord] =
                    inputWordlistBlocks[inputWord * blocksPerWord + inputBlock];
            //printf("%d => %d\n", inputWord * blocksPerWord + inputBlock, (outgoingDataCount * inputBlock) + inputWord);
        }
    }
}

void MFNHashTypePlain::setupActiveSaltArrays() {
    trace_printf("MFNHashTypePlain::setupActiveSaltArrays()\n");
    // Swap the hashes around into device format.  This will match the big or
    // little endianness of the device, obey the initial offset requested, 
    // and set the padding bit if needed.  Note that the locks must be set
    // before calling this - calling code must handle locking!


    uint32_t maximumSaltLength = 0; // Max salt length with padding in bytes.
    uint32_t saltWordsNeeded = 0; // Number of 32-bit words to contain longest salt.
    uint32_t saltWord; // uint32_t value to build each salt word in.
    uint32_t saltCount = 0;
    uint32_t i;
    uint32_t saltOffset = this->getSaltOffset(); // Offset for placing salts into the array.
    CHHashFileVSaltedDataBlob SaltData;


    std::vector<std::vector<uint8_t> >::iterator saltIt;

    // Clear all the salt arrays.
    this->saltLengths.clear();
    this->activeSalts.clear();
    this->activeSaltsDeviceformat.clear();
    this->activeIterationCounts.clear();

    // If this is a network client, try to update the salts over the network.
    if (this->CommandLineData->GetIsNetworkClient() && this->hashAttributes.hashUsesSalt) {
        MultiforcerGlobalClassFactory.getNetworkClientClass()->
            updateUncrackedSalts();
    }
    // Get the salts from the hashfile class.
    SaltData = this->HashFile->ExportUniqueSaltedData();

    // Store the active salt array.
    this->activeSalts = SaltData.SaltData;
    
    // If needed, store the iteration count
    if (this->hashAttributes.hashUsesIterationCount) {
        this->activeIterationCounts = SaltData.iterationCount;
    }
    

    if (this->getSetSaltPaddingBit()) {
        static_printf("Seting hash padding bit!\n");
    } else {
        static_printf("Not setting hash padding bit!\n");
    }

    static_printf("Salt offset: %d\n", this->getSaltOffset());

    /**
     * Iterate through the salts and do several things: Get the length of each
     * salt and push it into the array, add the padding bit to the end if
     * requested, and determine the maximum salt length in use for creating the
     * device length array
     */
    for (saltIt = this->activeSalts.begin(); saltIt < this->activeSalts.end();
            saltIt++) {
        
        // Set the length - this is the *actual* length of the salt, without
        // the padding bit or offset factored in.
        this->saltLengths.push_back((uint32_t)saltIt->size());

        // If a padding bit is requested, add it.
        if (this->getSetSaltPaddingBit()) {
            saltIt->push_back(0x80);
        }

        // Determine the maximum salt length.  We include the padding bit if
        // set - it will need space in the array!
        if (saltIt->size() > maximumSaltLength) {
            maximumSaltLength = saltIt->size();
        }
    }
    // If an offset is being used, add it in to increase the number of words.
    maximumSaltLength += saltOffset;
    static_printf("Maximum salt length: %d\n", maximumSaltLength);
    
    // Full words for the longest salt + padding + offset
    saltWordsNeeded = (maximumSaltLength / 4);
    // If there are more bytes, add another word.
    if (maximumSaltLength % 4) {
        saltWordsNeeded++;
    }
    static_printf("Max salt words: %d\n", saltWordsNeeded);

    // Resize the device salt vector as needed to contain all the salts.
    // This does waste space, but having constant lengths dramatically cleans
    // up the GPU and host side code for this...
    this->activeSaltsDeviceformat.clear();
    this->activeSaltsDeviceformat.resize(this->activeSalts.size() * saltWordsNeeded);

    // Iterate through the salts, setting things up as needed.
    for (saltIt = this->activeSalts.begin(); saltIt < this->activeSalts.end();
            saltIt++) {

        // Clear out the word being created.
        saltWord = 0;
        for (i = 0; i < saltIt->size(); i++) {
            // Set the byte in the saltWord.  This is offset by the saltOffset
            // value to put it in the right spot.
            if (!this->HashIsBigEndian) {
                // Hash is big endian.  Byte 0 is shifted 0.
                saltWord |= (uint32_t)saltIt->at(i) << (((i + saltOffset) % 4) * 8);
            } else {
                // Hash is little endian.  Byte 0 gets shifted << 24
                saltWord |= (uint32_t)saltIt->at(i) << (((3 - (i + saltOffset)) % 4) * 8);
            }
            // Check to see if the salt word needs to be pushed back.  This is
            // true if a word is full (%3 == 0), or if it is the last word in
            // the salt (== ->size() - 1).  In any case, push the word and reset
            // the storage value.
            if ((((i + saltOffset) % 4) == 3) ||
                    (i == (saltIt->size() - 1))) {
                if (((i + saltOffset) % 4) == 3) {
                    static_printf("Pushing for %4 - i: %d  soffset: %d\n", i, saltOffset);
                }
                if (i == (saltIt->size() - 1)) {
                    static_printf("Pushing for size\n");
                }
                // Offset in the salt array: (totalSalts * word + currentSalt)
                static_printf("1 Shoved saltword 0x%08x into position %d\n", saltWord,
                        this->activeSalts.size() * ((i + saltOffset) / 4) + saltCount);
                this->activeSaltsDeviceformat[this->activeSalts.size() *
                        ((i + saltOffset) / 4) + saltCount] = saltWord;
                saltWord = 0;
            }
        }
        saltCount++;
    }

    for (i = 0; i < this->activeSaltsDeviceformat.size(); i++) {
        static_printf("%d: 0x%08x\n", i, this->activeSaltsDeviceformat[i]);
    }
}

void MFNHashTypePlain::doAdditionalStaticDataSetup() {
    trace_printf("MFNHashTypePlain::doAdditionalStaticDataSetup()\n");

    this->MFNHashTypeSaltedMutex.lock();
    this->setupActiveSaltArrays();
    this->numberUniqueSalts = this->activeSalts.size();
    this->MFNHashTypeSaltedMutex.unlock();
}

void MFNHashTypePlain::doPerWorkunitDeviceSetup() {
    trace_printf("MFNHashTypePlain::doPerWorkunitDeviceSetup()\n");
    
    this->MFNHashTypeSaltedMutex.lock();

    this->setupActiveSaltArrays();
    this->copySaltArraysToDevice();
    this->numberUniqueSalts = this->activeSalts.size();

    this->MFNHashTypeSaltedMutex.unlock();
}
