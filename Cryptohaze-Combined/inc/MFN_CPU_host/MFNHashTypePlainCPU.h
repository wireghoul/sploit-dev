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

/**
 * @section DESCRIPTION
 *
 * MFNHashTypePlainCPU implements the CPU specific functions for plain hash
 * types.  This is a rough duplicate of functions in CHHashTypePlain for the
 * existing type.
 *
 * This class implements a subset of the device functionality needed for doing
 * CPU based SSE2 cracking.  Many of the functions are nulled out because they
 * are not needed - the CPUs can directly access the bitmaps/hash lists/etc.
 * 
 * The code is treated as far as the rest of the code goes as a kernel of vector
 * width (interleave * 4) * CPU_Threads - so a 4-thread instance with 3-way 
 * interleave would be (12 * 3) * 4 = 144-vector kernel.  The kernel handles
 * launching the threads such that they all run in parallel - the calling
 * code only specifies the number of threads to use.
 * 
 * The vector width is set by define - I'm not sure what the optimum vector
 * width for the CPUs is, so this lets us try different options.
 * 
 * Possibly smaller bitmaps would be good too - I will see what performance
 * looks like!
 */

#ifndef __MFNHASHTYPEPLAINCPU_H
#define __MFNHASHTYPEPLAINCPU_H

#include "MFN_Common/MFNHashTypePlain.h"
#include <boost/thread/mutex.hpp>
#include <boost/thread.hpp>
#include <boost/thread/barrier.hpp>


// How many steps per check-in period (with the barrier).  Used for performance
// reporting.  Note that this is *thread* steps, so will be multiplied by the
// total number of vectors.  This is sane enough for now.
#define MFNHASHTYPECPUPLAIN_STEPS_PER_PERIOD 1000000

/**
 * CPUSSEThreadData is used to pass data to each CPU thread.
 */
typedef struct CPUSSEThreadData {
    uint32_t threadNumber;
    uint64_t numberStepsTotal;
    boost::barrier *classBarrier;
} CPUSSEThreadData;

class MFNHashTypePlainCPU : public MFNHashTypePlain {
public:
    MFNHashTypePlainCPU(int hashLengthBytes);
    
    /**
     * Override base functionality with an actual add of devices.
     */
    int setCPUThreads(int numberCpuThreads);
    
protected:
    virtual void setupDevice();

    virtual void teardownDevice();

    virtual void allocateThreadAndDeviceMemory();

    virtual void freeThreadAndDeviceMemory();

    virtual void copyDataToDevice();

    virtual void copyStartPointsToDevice();
    
    virtual void setupClassForMultithreadedEntry();

    virtual void synchronizeThreads();

    virtual void setStartPoints(uint64_t perThread, uint64_t startPoint);

    virtual void copyDeviceFoundPasswordsToHost();

    // This needs to be here, as it requires the host success lists.
    virtual void outputFoundHashes();

    
    /**
     * Override the existing RunGPUWorkunit class - we do things rather
     * differently because we launch a bunch of threads in parallel instead
     * of doing just one thread at a time.  Because it is the CPU, there is no
     * need to launch segments of work - we can run the entire workunit in 
     * parallel.
     * 
     * @param WU Workunit to run.
     */
    virtual void RunGPUWorkunit(struct MFNWorkunitRobustElement *WU);

    virtual void cpuSSEThread(CPUSSEThreadData threadData) = 0;
    
    // Salted hashes need this.  Right now, we don't have it.
    void copySaltArraysToDevice() { };
    
    /**
     * A pointer to the host start point address
     */
    uint8_t *HostStartPointAddress;
    
    /**
     * The number of CPU threads to run.
     */
    uint16_t numberCPUThreads;
    
    /**
     * A container for the CPU threads launched.
     */
    std::vector<boost::thread *> CPUThreadObjects;
    
    /**
     * A container for the per-thread run data.
     */
    std::vector<CPUSSEThreadData> CPUThreadData;

    /**
     * A 512kb bitmap for a-value checking.
     */
    std::vector<uint8_t> classBitmapLookup_a;

    /**
     * Mutex for HashTypePlainCPU!  Mostly for hash reporting.
     */
    boost::mutex MFNHashTypePlainCPUMutex;
    
};

#endif