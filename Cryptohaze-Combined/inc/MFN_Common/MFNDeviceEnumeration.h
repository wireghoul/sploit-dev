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

#ifndef __MFNDEVICEENUMERATION_H
#define __MFNDEVICEENUMERATION_H

/**
 *
 * @section DESCRIPTION
 *
 * This class handles the fact that on a system with a number of OpenCL drivers
 * installed (and CUDA drivers), a device will show up numerous times.  This
 * class attempts to resolve this by creating a unique list of devices with the
 * supported methods of interacting with them (as well as exporting this list
 * to the central server if requested).  The general methods of interacting
 * with a device:
 * nVidia GPU:
 *  -- CUDA
 *  -- nVidia OpenCL
 *  -- Apple OpenCL
 * AMD/ATI GPU:
 *  -- AMD OpenCL
 *  -- Apple OpenCL (?)
 * Host CPU:
 *  -- Native (host code)
 *  -- Intel OpenCL
 *  -- AMD OpenCL
 *  -- Apple OpenCL
 * 
 * The goal is to disambiguate devices, and pick the best method of interfacing
 * with each device. 
 * 
 * UNFORTUNATELY, there is no good way to pick out specific devices to match
 * them, because (currently) OpenCL does not provide a way to get the PCI bus
 * and device IDs like CUDA does.  So, some "loose approximations" are done - 
 * if the same number of identically named devices are present for the different
 * platforms, they are considered present on both.  I'm sure this will break
 * in some goofball edge case, but it's good enough for now - most people
 * won't be running weird mixes of hardware trying to break this (I hope...)
 */

#include <vector>
#include <string>
#include <stdint.h>

// Device type defines for this class.
#define MFN_DEVICE_TYPE_CPU 0x10
#define MFN_DEVICE_TYPE_GPU 0x11

// Define what to add - in case someone is not using OpenCL or CUDA
#define MFN_DEVICE_ENUMERATION_CUDA 1
#define MFN_DEVICE_ENUMERATION_OPENCL 1

/**
 * A structure to contain an OpenCL "node ID" - platform & device.  Will
 * 4B devices in a system be enough?
 */
typedef struct MFNDeviceInformationOpenCL {
    uint32_t openCLPlatform;
    uint32_t openCLDevice;
};
/**
 * A structure containing per-platform information for OpenCL
 */
typedef struct MFNPlatformInformationOpenCL {
    uint32_t openCLPlatformId;
    std::string openCLPlatformName;
    std::string openCLPlatformVendor;
    std::string openCLPlatformVersion;
};


/**
 * A structure to contain the device information.  Unlike the other device
 * structure in use, this is better suited to describing the device and its
 * attributes.
 */
typedef struct MFNDeviceInformation {
    // Unique device ID in this arrangement.
    uint32_t deviceId;
    // Name of the actual device from CUDA/OpenCL
    std::string deviceName;
    // Name of the OpenCL Platform, if present
    std::string openCLPlatformName;
    // Device type - MFN_DEVICE_CPU, MFN_DEVICE_GPU
    uint8_t deviceType;
    // Total device memory in bytes.  If different platforms give different
    // readings, just pick one.
    uint64_t deviceTotalMemoryBytes;
    // Total local/shared memory in bytes.  For a CPU, this is 0.
    uint32_t deviceLocalMemoryBytes;
    // Device number cores - CUDA cores, AMD compute units, CPU cores.
    uint32_t deviceNumberCores;
    // Set to 1 if the device is CUDA capable, else 0.
    uint8_t isCudaCapable;
    // CUDA device ID (or one of the valid ones for this device type).  As CUDA
    // is only on a platform once, it can only be one CUDA device ID
    uint32_t cudaDeviceId;
    // Set to 1 if the device appears in at least 1 OpenCL platform, else 0.
    uint8_t isOpenCLCapable;
    // Ways to find this device via OpenCL - it may be present in multiple
    // platforms (such as Intel OpenCL & AMD OpenCL)
    std::vector<MFNDeviceInformationOpenCL> openCLAccess;
    // Set to 1 if the device is accessible as a "host CPU" - even if it is
    // also present in OpenCL.
    uint8_t isHostCodeCapable;
    // PCI Bus (if available)
    uint16_t pciBus;
    // PCI device (if available)
    uint16_t pciDevice;
    
    // TODO: More options!
    
};




class MFNDeviceEnumeration {
public:
    MFNDeviceEnumeration();
    ~MFNDeviceEnumeration();
    
    /**
     * Attempts to enumerate all the devices on the system.
     */
    void EnumerateAllDevices();
    
    /**
     * Gets a vector of unique devices in the system.  This can be used to 
     * determine how to set up the actual runtime environment.
     * 
     * @return A vector of device structs that should be unique.
     */
    std::vector<MFNDeviceInformation> GetUniqueDevices();
    
    /**
     * Exports the current data in a protobuf.
     * 
     * @param exportData Pointer to a string to fill with the data.
     */
    void ExportDevicesToRemoteSystem(std::string * exportData);
    
    /**
     * Imports a protobuf string from a remote system and populates the internal
     * structure, overwriting other data.
     * 
     * @param importData The string to import from.
     */
    void ImportDevicesFromRemoteSystem(std::string & importData);
    
private:
    // A list of all the unique devices, merged together.
    std::vector<MFNDeviceInformation> uniqueDevices;
    
    // A list of all the CUDA-enumerated devices
    std::vector<MFNDeviceInformation> CUDADevices;
    
    // A list of the current OpenCL platforms
    std::vector<MFNPlatformInformationOpenCL> openCLPlatforms;
    
    // A list of all the OpenCL Devices gained from different platforms
    std::vector<MFNDeviceInformation> openCLDevices;
    
    // A struct containing the information about the host CPU.
    MFNDeviceInformation hostCPUInformation;
    
    /**
     * Enumerates all the CUDA devices in the system.
     */
    void enumerateCUDADevices();
    
    /**
     * Enumerates all the OpenCL platforms and devices in the system.
     */
    void emumerateOpenCLDevices();
    
    /**
     * Tries to find out things about the host CPU from the platform-specific
     * methods.
     */
    void getHostCPUInformation();
    
    /**
     * Merges all the collected data into a coherent whole.
     */
    void mergeAllDevices();
    
};



#endif