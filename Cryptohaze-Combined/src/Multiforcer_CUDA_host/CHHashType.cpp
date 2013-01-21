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

/*
 * CHHashType.cpp implements the basic functionality used for hash
 * cracking classes.  Very little in here should need to be overridden.
 *
 *
 */

#include "Multiforcer_CUDA_host/CHHashType.h"
#include "Multiforcer_Common/CHCommon.h"

// Include the various classes
#include "Multiforcer_Common/CHDisplay.h"
#include "Multiforcer_CUDA_host/CHCommandLineData.h"

// Default constructor.  Initialize all values to sane values.
CHHashType::CHHashType() {
    // Set all the class pointers to NULL to indicate they are not set.
    this->CommandLineData = NULL;
    this->Charset = NULL;
    this->Workunit = NULL;
    this->HashFile = NULL;
    this->Display = NULL;

    // Initialize Multithreading data to all null
    this->ActiveThreadCount = 0;
    memset(this->ThreadData, 0, MAX_SUPPORTED_THREADS * sizeof(threadRunData));

#if USE_BOOST_THREADS
    memset(this->ThreadObjects, 0, MAX_SUPPORTED_THREADS * sizeof(boost::thread *));
#else
    memset(this->ThreadIds, 0, MAX_SUPPORTED_THREADS * sizeof(pthread_t));
#endif

}


// Add a GPU deviceID to the list of active devices.
// Returns 0 on failure (probably too many threads), 1 on success.
int CHHashType::addGPUDeviceID(int deviceId) {

    // Ensure the CommandLineData class has been added.
    // This is likely a programmer error, not a runtime error.
    // Therefore, exit.
    if (!this->CommandLineData) {
        if (this->Display) {
            sprintf(this->statusBuffer, "Must add command line data first!");
            this->Display->addStatusLine(this->statusBuffer);
        } else {
            printf("Must add command line data first!\n");
        }
        exit(1);
    }

    // If the device ID is invalid, reject.
    // This is likely a runtime error, so just return 0.
    if (deviceId >= this->CommandLineData->GetCUDANumberDevices()) {
        if (this->Display) {
            sprintf(this->statusBuffer, "Invalid device ID %d", deviceId);
            this->Display->addStatusLine(this->statusBuffer);
        } else {
            printf("Invalid device ID %d", deviceId);
        }
        return 0;
    }

    // If all threads are full, do not add the thread.
    // Again, a runtime error, return 0.
    if (this->ActiveThreadCount >= (MAX_SUPPORTED_THREADS - 1)) {
        if (this->Display) {
            sprintf(this->statusBuffer, "Too many active threads!");
            this->Display->addStatusLine(this->statusBuffer);
        } else {
            printf("Too many active threads!");
        }
        return 0;
    }

    // Set up the GPU thread.
    this->ThreadData[this->ActiveThreadCount].valid = 1;
    this->ThreadData[this->ActiveThreadCount].gpuThread = 1;
    this->ThreadData[this->ActiveThreadCount].gpuDeviceId = deviceId;
    this->ThreadData[this->ActiveThreadCount].threadID =
            this->ActiveThreadCount;
    this->ThreadData[this->ActiveThreadCount].HashType = this;


    // Set up the device parameters
    // If they are manually forced, set them, else set them to defaults.
    // CUDA Blocks
    if (this->CommandLineData->GetCUDABlocks()) {
        this->ThreadData[this->ActiveThreadCount].CUDABlocks =
            this->CommandLineData->GetCUDABlocks();
    } else {
        this->ThreadData[this->ActiveThreadCount].CUDABlocks =
            getCudaDefaultBlockCountBySPCount(getCudaStreamProcessorCount(deviceId));
    }
    // CUDA Threads
    if (this->CommandLineData->GetCUDAThreads()) {
        this->ThreadData[this->ActiveThreadCount].CUDAThreads =
            this->CommandLineData->GetCUDAThreads();
    } else {
        this->ThreadData[this->ActiveThreadCount].CUDAThreads =
            getCudaDefaultThreadCountBySPCount(getCudaStreamProcessorCount(deviceId));
    }
    // Default execution time
    if (this->CommandLineData->GetTargetExecutionTimeMs()) {
        this->ThreadData[this->ActiveThreadCount].kernelTimeMs =
            this->CommandLineData->GetTargetExecutionTimeMs();
    } else {
        if (getCudaHasTimeout(deviceId)) {
            this->ThreadData[this->ActiveThreadCount].kernelTimeMs =
                DEFAULT_CUDA_EXECUTION_TIME;
        } else {
            this->ThreadData[this->ActiveThreadCount].kernelTimeMs =
                DEFAULT_CUDA_EXECUTION_TIME_NO_TIMEOUT;
        }
    }
    


    // Increment the active thread count.
    this->ActiveThreadCount++;

    // If the display unit is active, add a notice.
    if (this->Display) {
        sprintf(this->statusBuffer, "Added GPU device %d", deviceId);
        this->Display->addStatusLine(this->statusBuffer);
    }
    return 1;
}

// Add a CPU thread to execute.
// Returns 0 on failure, 1 on success.
int CHHashType::addCPUThread() {

    // If all threads are full, do not add the thread.
    // Again, a runtime error, return 0.
    if (this->ActiveThreadCount >= (MAX_SUPPORTED_THREADS - 1)) {
        if (this->Display) {
            sprintf(this->statusBuffer, "Too many active threads!");
            this->Display->addStatusLine(this->statusBuffer);
        } else {
            printf("Too many active threads!");
        }
        return 0;
    }

    // Set up the CPU thread.
    this->ThreadData[this->ActiveThreadCount].valid = 1;
    this->ThreadData[this->ActiveThreadCount].cpuThread = 1;
    this->ThreadData[this->ActiveThreadCount].threadID =
            this->ActiveThreadCount;
    this->ThreadData[this->ActiveThreadCount].HashType = this;
    // Increment the active thread count.
    this->ActiveThreadCount++;

    // If the display unit is active, add a notice.
    if (this->Display) {
        sprintf(this->statusBuffer, "Added CPU thread");
        this->Display->addStatusLine(this->statusBuffer);
    }
    return 1;
}


// Set the various subclasses that are needed.
// You should NOT need to override these!
void CHHashType::setCommandLineData(CHCommandLineData *NewCommandLineData) {
    this->CommandLineData = NewCommandLineData;
}
void CHHashType::setCharset(CHCharset *NewCharset) {
    this->Charset = NewCharset;
}
void CHHashType::setWorkunit(CHWorkunitBase *NewWorkunit) {
    this->Workunit = NewWorkunit;
}
void CHHashType::setHashFile(CHHashFileTypes *NewHashFile) {
    this->HashFile = NewHashFile;
}
void CHHashType::setDisplay(CHDisplay *newDisplay) {
    this->Display = newDisplay;
}

