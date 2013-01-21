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

#ifndef __MFNHASHCLASSLAUNCHER_H__
#define __MFNHASHCLASSLAUNCHER_H__

/**
 * @section DESCRIPTION
 *
 * This class implements the launcher for the hash class threads.
 * 
 * This class is used to manage the multithreading for the different thread
 * types - CUDA, OpenCL, CPU.  The calling code specifies the type of hash, 
 * and the type of thread to add.  This handles the rest!
 * 
 * Note that not all hash types will have all thread types implemented - if
 * a thread type is not implemented, or the device specification is invalid,
 * the creator will return false.
 */





#include <stdint.h>
#include <vector>
#include <boost/thread.hpp>
#include "MFN_Common/MFNHashIdentifiers.h"
#include "MFN_Common/MFNCommandLineData.h"



class MFNHashType;

typedef struct ClassLauncherData {
    MFNHashType *HashTypeClass;
    int passwordLength;
    int threadID;
} ClassLauncherData;

class MFNHashClassLauncher {
public:
    /**
     * Standard constructor.
     */
    MFNHashClassLauncher();
    
    /**
     * Standard destructor.
     */
    ~MFNHashClassLauncher();
    
    
    /**
     * Sets the hash type to create threads for.
     * 
     * @param newHashType The hash ID to create threads for.
     * @return True on success, false on invalid hash ID.
     */
    int setHashType(uint32_t newHashType);
    
    /**
     * Adds a CPU class to the execution vector.
     * @return True if the thread was added successfully, else false.
     * 
     * @param numberCPUThreads The number of CPU threads to add.
     */
    bool addCPUThreads(uint16_t numberCPUThreads);
    
    /**
     * Adds a CUDA thread to the execution vector.
     * @param newCudaDeviceId The CUDA device to use.
     * @return True if success, false if invalid device ID or no type available.
     */
    bool addCUDAThread(uint16_t newCudaDeviceId);
    
    bool addOpenCLThread(uint16_t newOpenCLPlatform, uint16_t newOpenCLDevice);
   
    /**
     * Launches the threads with the given password length.
     * @param passwordLength
     * @return 
     */
    bool launchThreads(uint16_t passwordLength);
    
    /**
     * Returns a pointer to a specific class in the array.  Not sure why you'd 
     * need this, but it's available.
     * 
     * @param classId Class element to return a pointer for.
     * @return Pointer to the class, else NULL if an invalid index.
     */
    MFNHashType *getClassById(uint16_t classId);
    
    /**
     * Adds all devices from the command line data device list.
     * 
     * @param A vector of MFNDeviceInformation types to add.
     * @return True if they were all added successfully, else false.
     */
    bool addAllDevices(std::vector<MFNDeviceInformation>);
    
private:
    /**
     * Container for all the classes created.
     */
    std::vector<MFNHashType *> ClassVector;
    
    /**
     * Container for the launched thread objects.
     */
    std::vector<boost::thread *> ThreadObjects;
    
    uint32_t HashType;
};


#endif