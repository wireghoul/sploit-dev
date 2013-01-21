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

// Various CUDA util functions


#ifndef _CHCUDAUTILS_H
#define _CHCUDAUTILS_H

#include "CH_Common/CHCommonBuildIncludes.h"


#include <cuda.h>
#include <cuda_runtime_api.h>

class CHCUDAUtils {
public:
    /**
     * Prints the basic information about the chosen device.
     * 
     * This function will query a given CUDA device ID and print out the full
     * relevant information about it.
     * 
     * @param CUDADeviceId
     */
    void PrintCUDADeviceInfo(int CUDADeviceId);
    
    /**
     * Copy of the function in shrUtils.h to get the number of cores per MP.
     * 
     * This function will return the number of cores per MP, allowing for correct
     * conversions to the core count on the device.  It's nice to display
     * the correct information to people...
     * 
     * @param major The CUDA Capability major version
     * @param minor The CUDA Capability minor version
     * @return The number of CUDA cores per multiprocessor
     */
    int ConvertSMVer2Cores(int major, int minor);
    
    /**
     * Return the number of stream processors for the given CUDA device ID.
     * 
     * @param CUDADeviceId
     * @return The number of SPs in this CUDA device.
     */
    int getCudaStreamProcessorCount(int CUDADeviceId);
    
    /**
     * Returns true if the device has a timeout (display attached), else false.
     * 
     * If a CUDA device has a display attached, there will be a timeout associated
     * with it.  If not, we can run longer kernels.
     * 
     * @param CUDADeviceId
     * @return True if there is a timeout, else false.
     */
    int getCudaHasTimeout(int CUDADeviceId);
 
    /**
     * Returns true if the device is a Fermi class device (SM20 or SM21).
     * 
     * @param CUDADeviceId The CUDA device ID to test
     * @return True if a Fermi, else false.
     */
    int getCudaIsFermi(int CUDADeviceId);
    
    /**
     * Returns the default thread count for the specified device.
     * 
     * @param CUDADeviceId
     * @return The default thread count for the specified device.
     */
    int getCudaDefaultThreadCount(int CUDADeviceId);
    
    /**
     * Returns the default block count for the specified device.
     * 
     * @param CUDADeviceId
     * @return The default block count for the specified device.
     */
    int getCudaDefaultBlockCount(int CUDADeviceId);
    
    /**
     * Returns the number of bytes of global memory on the device.
     * 
     * @param CUDADeviceId
     * @return Amount of global memory on the device.
     */
    uint64_t getCudaDeviceGlobalMemory(int CUDADeviceId);
    
    /**
     * Returns the number of bytes of shared memory on the device.  Currently,
     * this is either 16kb or 48kb.
     * 
     * @param CUDADeviceId
     * @return Bytes of shared memory on the device.
     */
    uint32_t getCudaDeviceSharedMemory(int CUDADeviceId);
    
    /**
     * Returns true if the device is integrated  & shares host RAM.
     * 
     * @param CUDADeviceId
     * @return True if the device is integrated & shares host RAM.
     */
    int getCudaIsIntegrated(int CUDADeviceId);
    
    /**
     * Returns true if the device is capable of mapping host memory.
     * 
     * @param CUDADeviceId
     * @return True if capable of mapping host mem.
     */
    int getCudaCanMapHostMemory(int CUDADeviceId);
    
    
    /**
     * Get the number of CUDA devices present in the system.
     * @return 
     */
    int getCudaDeviceCount();
    
    
    
private:
    /**
     * Returns a cudaDeviceProp struct for the device ID.
     * 
     * @param CUDADeviceId
     * @return A filled out cudaDeviceProp struct, or totally cleared if ID not valid.
     */
    cudaDeviceProp getDevicePropStruct(int CUDADeviceId);

    /**
     * Returns true if the device ID is valid on the system, else false.
     * 
     * @param CUDADeviceId
     * @return 
     */
    int getIsValidDeviceId(int CUDADeviceId);
};


#endif
