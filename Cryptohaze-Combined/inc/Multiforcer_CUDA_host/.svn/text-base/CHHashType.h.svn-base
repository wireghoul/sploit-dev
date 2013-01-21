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

#ifndef __CHHASHTYPE_H
#define __CHHASHTYPE_H

/*
 * CHHashType is the basic hash cracking class.  This implements the basic
 * interfaces used, and some code that probably will not need to change
 * for different classes.  Modifications here should not be needed!  If you
 * need different functionality, please implement it by overriding
 * the functionality in the subclass.
 */

#include "Multiforcer_Common/CHCommon.h"

// Maximum supported GPU/CPU thread count.  Updating this will require
// updating the display code to scroll the rates.
#define MAX_SUPPORTED_THREADS 16

// Forward declare required classes
class CHHashType;
class CHCommandLineData;
class CHCharset;
class CHWorkunitBase;
class CHHashFileTypes;
class CHDisplay;

// Runtime data passed to each CPU/GPU thread.
typedef struct threadRunData {
    char valid;         // Nonzero to indicate a valid unit.
    char cpuThread;     // Nonzero if a CPU thread
    char gpuThread;     // Nonzero if a GPU thread
    char gpuDeviceId;   // CUDA device ID
    int  threadID;      // Thread ID - for identification/structures.
    int  CUDAThreads;   // CUDA thread count for GPUs
    int  CUDABlocks;    // CUDA block count for GPUs
    int  kernelTimeMs;  // Target execution time for GPUs
    CHHashType *HashType;   // Copy of the invoking class to reenter.
 } threadRunData;

class CHHashType {
public:
    CHHashType(); // Default constructor

    // Add a GPU deviceID to the list of active devices.
    // Returns 0 on failure (probably too many threads), 1 on success.
    virtual int addGPUDeviceID(int deviceId);

    // Add a CPU thread to execute.
    // Returns 0 on failure, 1 on success.
    virtual int addCPUThread();

    // Main entry point for cracking.
    virtual void crackPasswordLength(int passwordLength) = 0;
    
    // Thread to run on the GPU.  Entered by the pthread entry call.
    virtual void GPU_Thread(void *data) = 0;

    // Set the various subclasses that are needed.
    // You should NOT need to override these!
    virtual void setCommandLineData(CHCommandLineData *NewCommandLineData);
    virtual void setCharset(CHCharset *NewCharset);
    virtual void setWorkunit(CHWorkunitBase *NewWorkunit);
    virtual void setHashFile(CHHashFileTypes *NewHashFile);
    virtual void setDisplay(CHDisplay *newDisplay);
    //virtual void setHashFile(CHHashFileMSSQL *NewHashFile);
protected: // This needs to be accessible to derived classes.
    // Classes needed for getting various bits of runtime data.
    // These must be set by the invoking code before being used.
    CHCommandLineData *CommandLineData;
    CHCharset *Charset;
    CHWorkunitBase *Workunit;
    CHHashFileTypes *HashFile;
    CHDisplay *Display;

    // Multithreading data.
    int ActiveThreadCount;
    threadRunData ThreadData[MAX_SUPPORTED_THREADS];

#if USE_BOOST_THREADS
    boost::thread *ThreadObjects[MAX_SUPPORTED_THREADS];
#else
    pthread_t ThreadIds[MAX_SUPPORTED_THREADS];
#endif

    // Buffer for use passing data to the display
    char statusBuffer[1000];
    
};

#endif