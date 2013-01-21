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
 * MFNHashTypePlainCUDA implements the CUDA specific functions for plain hash
 * types.  This is a rough duplicate of functions in CHHashTypePlain for the
 * existing type.
 *
 * This contains two different types of storage for host vs device values.
 *
 * The host pointers are stored as pointers most of the time.  This allows
 * for the use of cudaHostAlloc to get pinned memory to speed transfers
 * as needed.
 *
 * The vectors get expanded to their needed size by each thread if they are too
 * small, otherwise they remain the same size.
 */

#ifndef __MFNHASHTYPEPLAINCUDA_H
#define __MFNHASHTYPEPLAINCUDA_H

#include "MFN_Common/MFNHashTypePlain.h"

class MFNHashTypePlainCUDA : public MFNHashTypePlain {
public:
    MFNHashTypePlainCUDA(int hashLengthBytes);
    
    /**
     * Override base functionality with an actual add of devices.
     */
    int setCUDADeviceID(int newCUDADeviceId);
    
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
    
    virtual void copyWordlistToDevice(std::vector <uint8_t> &wordlistLengths,
        std::vector<uint32_t> &wordlistData);
    
    // Copy wordlist sizes to the device - location is kernel dependent.
    virtual void copyWordlistSizeToDevice(uint32_t wordCount, uint8_t blocksPerWord) { };

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

    /**
     * Device full hashlist pointer.
     *
     * This contains the device memory address in which the
     * device hast list is stored.  This is used for the device hashlist
     * allocation, copy, and free.
     */
    uint8_t *DeviceHashlistAddress;

    /**
     * A pointer to the host found password array region.
     */
    uint8_t *HostFoundPasswordsAddress;

    /**
     * The device found password address.
     */
    uint8_t *DeviceFoundPasswordsAddress;

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
    uint8_t *DeviceSuccessAddress;

    /**
     * Pointers to the device bitmaps.  Null if not present.
     */
    uint8_t *DeviceBitmap128mb_a_Address;
    uint8_t *DeviceBitmap128mb_b_Address;
    uint8_t *DeviceBitmap128mb_c_Address;
    uint8_t *DeviceBitmap128mb_d_Address;

    uint8_t *DeviceBitmap256kb_Address;

    /**
     * Pointer to device start point addresses
     */
    uint8_t *DeviceStartPointAddress;
    
    /**
     * Pointer to the device start passwords
     */
    uint8_t *DeviceStartPasswords32Address;

    /**
     * A pointer to the host start point address
     */
    uint8_t *HostStartPointAddress;
    
    /*
     * Wordlist data - lengths, data.  The rest will be a kernel arg.
     */
    uint32_t *DeviceWordlistBlocks;
    uint8_t *DeviceWordlistLengths;


    /**
     * Variable set if we are using zero copy.  Zero copy will be used if the
     * user requests it, or if the device is an integrated device sharing
     * host memory (as there is no benefit to copying).
     */
    uint8_t useZeroCopy;

    // Salted stuff
    
    // The number of salts in the array copied to the device.
    // May be different per-thread.
    uint32_t numberSaltsCopiedToDevice;

    uint8_t *DeviceSaltLengthsAddress; // Address of device salt lengths.
    uint8_t *DeviceSaltValuesAddress; // Address of device salt array
    
    
};

#endif