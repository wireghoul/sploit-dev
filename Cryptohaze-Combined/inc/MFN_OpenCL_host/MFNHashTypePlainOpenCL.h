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
 * MFNHashTypePlainOpenCL implements the OpenCL specific functions for plain hash
 * types.  This is a rough duplicate of functions in CHHashTypePlain for the
 * existing type.
 */

#ifndef __MFNHASHTYPEPLAINOPENCL_H
#define __MFNHASHTYPEPLAINOPENCL_H

#include "MFN_Common/MFNHashTypePlain.h"

#include "OpenCL_Common/GRTOpenCL.h"

class MFNHashTypePlainOpenCL : public MFNHashTypePlain {
public:
    MFNHashTypePlainOpenCL(int hashLengthBytes);
    ~MFNHashTypePlainOpenCL();
    
    /**
     * Override base functionality with an actual add of devices.
     */
    int setOpenCLDeviceID(int newOpenCLPlatformId, int newOpenCLDeviceId);
    
protected:
    virtual void setupDevice();

    virtual void teardownDevice();
        
    virtual void doKernelSetup();

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
    
    virtual void copyWordlistToDevice(std::vector <uint8_t> &wordlistLengths,
        std::vector<uint32_t> &wordlistData);
    
    // Copy wordlist sizes to the device - location is kernel dependent.
    virtual void copyWordlistSizeToDevice(cl_uint wordCount, cl_uchar blocksPerWord) { };
    
    /**
     * Return the kernel to run.  This allows for hash types with multiple
     * kernels to override the default behavior and return the correct kernel
     * so the wordlist can be copied properly and arguments set.
     */
    virtual cl_kernel getKernelToRun() {
        return this->HashKernel;
    }

    
    /**
     * Copy the salt arrays specifically to the device.
     */
    virtual void copySaltArraysToDevice();
    
    /**
     * This copies the salt data to the individual device.  This is done by
     * the hash leaf node as it depends on the name of the hash function.
     */
    virtual void copySaltConstantsToDevice() { };
    
    // Host and memory device addresses.  These are per-class now, instead
    // of a vector.  This should make things easier and more foolproof.

    CryptohazeOpenCL *OpenCL;

    /**
     * Device full hashlist pointer.
     *
     * This contains the device memory address in which the
     * device hast list is stored.  This is used for the device hashlist
     * allocation, copy, and free.
     */
    cl_mem DeviceHashlistAddress;

    /**
     * A pointer to the host found password array region.
     */
    uint8_t *HostFoundPasswordsAddress;

    /**
     * The device found password address.
     */
    cl_mem DeviceFoundPasswordsAddress;

    /**
     * Pointer containing the host success/found password flags
     */
    uint8_t *HostSuccessAddress;

    /**
     * Pointer containing the host success reported flags.
     */
    uint8_t *HostSuccessReportedAddress;

    /**
     * Pointer containing the devices success addresses
     */
    cl_mem DeviceSuccessAddress;

    /**
     * Pointers to the device bitmaps.  Null if not present.
     */
    cl_mem DeviceBitmap128mb_a_Address;
    cl_mem DeviceBitmap128mb_b_Address;
    cl_mem DeviceBitmap128mb_c_Address;
    cl_mem DeviceBitmap128mb_d_Address;

    cl_mem DeviceBitmap8kb_Address;
    cl_mem DeviceBitmap16kb_Address;
    cl_mem DeviceBitmap32kb_Address;
    

    cl_mem DeviceBitmap256kb_a_Address;
    
    /**
     * Pointer to device start point addresses
     */
    cl_mem DeviceStartPointAddress;

    /**
     * Pointer to the device start passwords
     */
    cl_mem DeviceStartPasswords32Address;

    /**
     * A pointer to the host start point address
     */
    uint8_t *HostStartPointAddress;
    
    cl_mem DeviceForwardCharsetAddress;
    cl_mem DeviceReverseCharsetAddress;
    cl_mem DeviceCharsetLengthsAddress;
    
    /*
     * Wordlist data - lengths, data.  The rest will be a kernel arg.
     */
    cl_mem DeviceWordlistBlocks;
    cl_mem DeviceWordlistLengths;
    
    cl_program HashProgram;
    cl_kernel HashKernel;
    
    // Vector of kernels in the event of a type with multiple kernels.
    std::vector<cl_kernel> HashKernelVector;
    
    // As OpenCL is "late binding" with regards to memory, store the total
    // mem on small devices and figure out how much we can access.
    cl_ulong DeviceAvailableMemoryBytes;
    
    // The shared bitmap size, in kb.
    int sharedBitmapSize;

    // The number of salts in the array copied to the device.
    // May be different per-thread.
    uint32_t numberSaltsCopiedToDevice;

    cl_mem DeviceSaltLengthsAddress; // Address of device salt lengths.
    cl_mem DeviceSaltValuesAddress; // Address of device salt array
    
    cl_mem DeviceIterationCountAddresses; // Addresses of the iteration count
    
    // Temporary space that is allocated but not set.  The kernels must set
    // this themselves...
    cl_mem DeviceTempSpaceAddress;

};

#endif