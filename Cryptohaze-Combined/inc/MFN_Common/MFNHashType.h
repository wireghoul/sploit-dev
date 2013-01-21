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
 * The class MFNHashType is the reimplementation of CHHashType for
 * support of multiple platforms.  All the device-specific functionality
 * is to be in subclasses for the platform type and hash type.
 * 
 * The base class and general derived classes (MFNHashTypePlain, etc) will be
 * purely platform agnostic.  ALL the device functionality MUST be contained
 * within subclasses!
 * 
 * This allows for more flexibility with pre-processing hashes for improved
 * performance and for pushing more and more varied data to the device for 
 * cracking to support algoriths such as IKE, WPA, and file cracking.
 * 
 * This class can only be used for one type of device at a time - CPU, CUDA, 
 * or OpenCL.  If multiple devices are needed, the calling code must make 
 * multiple versions of this class with the required devices set up for each 
 * instance.
 *
 * Static variables are used for data that will be across all classes.  This
 * goes up to the HashTypePlain type classes - the device-specific classes
 * will have their own data.
 * 
 * Class derivation is as follows:
 * 
 * MFNHashType: The very base class for all hash types, all devices.  This
 * keeps the basic functionality, adding devices, and the basic device functions
 * that must be implemented by all classes as virtual functions.
 * 
 * MFNHashTypePlain: This class implements a generic plain hash type.  This
 * includes getting the hashes, running the kernels (generically), etc.  This
 * does not have any device specific code.
 * 
 * MFNHashTypePlainCUDA: This class implements CUDA specific functions for
 * the plan hash type - allocating memory, freeing memory, transferring data
 * to the device, etc.
 * 
 * MFNHashTypePlainCUDAMD5: This is the final leaf class node, and implements
 * the MD5 specific kernels/calls/etc to do MD5 cracking with CUDA.
 */

#ifndef __MFNHASHTYPE_H
#define __MFNHASHTYPE_H

#include "Multiforcer_Common/CHCommon.h"

// Forward declare some classes
class MFNHashType;
class MFNCommandLineData;
class CHCharsetNew;
class MFNWorkunitBase;
class CHHashFileV;
class MFNDisplay;

/**
 * Declare some types for threads - CUDA, CPU, OpenCL.
 */
#define MFN_HASH_TYPE_CUDA_THREAD    0x10
#define MFN_HASH_TYPE_OPENCL_THREAD  0x11
#define MFN_HASH_TYPE_CPU_THREAD     0x12
#define MFN_HASH_TYPE_INVALID_DEVICE 0xff

/**
 * This structure is used to pass around thread specific data to the various
 * executing threads while they are within the HashTypeV class and subclasses.
 * This contains all the needed information to set up the devices and run the
 * tasks.  Note that much of the data is agnostic between CUDA and OpenCL - 
 * this is by design, as they have very similar concepts.
 */
//typedef struct MFNThreadRunData {
//    uint8_t  valid;         /**< Non-zero if this is a valid set of data. */
//    uint8_t  cpuThread;     /**< Non-zero if this is a CPU thread (SSE2). */
//    uint8_t  gpuThreadCUDA; /**< Non-zero if this is a CUDA thread */
//    uint8_t  gpuThreadOpenCL; /**< Non-zero if this is an OpenCL thread. */
//    uint8_t  openCLPlatformId; /**< OpenCL Platform ID if is OpenCL */
//    uint8_t  gpuDeviceId;   /**< CUDA or OpenCL device ID */
//    uint16_t threadID;      /**< Current thread ID (unique) */
//    uint16_t GPUThreads;   /**< CUDA or OpenCL "thread" count */
//    uint16_t GPUBlocks;    /**< CUDA or OpenCL "block" count */
//    uint16_t kernelTimeMs;  /**< Target execution time for GPUs */
//    MFNHashType *HashTypeV;   // Copy of the invoking class to reenter.
//} MFNThreadRunData;

class MFNHashType {
public:
    /**
     * Default constructor for CHHashTypeV.
     */
    MFNHashType();
    /**
     * Sets the CUDA device for this class.
     * 
     * This function sets the CUDA device to have code running on it.  It will
     * return 1 on success, or 0 if there is no such device or another issue.
     * It will also return 0 if the device type is not supported by the
     * current class.
     * 
     * @param newCUDADeviceId The CUDA device ID to use
     * @param newThreadId The unique thread ID to use
     * @return 1 on success, 0 on failure.
     */
    virtual int setCUDADeviceID(int newCUDADeviceId) {
        return 0;
    }
    
    /**
     * Sets the OpenCL device for the class
     * 
     * This function sets the OpenCL device (CPU or GPU, prefer GPU) to have
     * code running.  It will return 1 on success, 0 on failure.
     * It will also return 0 if the device type is not supported by the
     * current class.
     * 
     * @param newOpenCLPlatformId The OpenCL platform ID to use for this device.
     * @param newOpenCLDeviceId The OpenCL device ID to use.
     * @param newThreadId The unique thread ID to use
     * @return 1 on success, 0 on failure.
     */
    virtual int setOpenCLDeviceID(int newOpenCLPlatformId, int newOpenCLDeviceId) {
        return 0;
    }

    /**
     * This adds a number of CPU threads to participate in the cracking.
     * 
     * This function will add the specified number of CPU threads to the 
     * cracking task.  This should be determined previously or specified by
     * the user.  It will also return 0 if the device type is not supported 
     * by the current class.
     * 
     * @param numberCpuThreads How many CPU threads to spawn off.
     * @param newThreadId The unique thread ID to use
     * @return 1 on success, 0 on failure.
     */
    virtual int setCPUThreads(int numberCpuThreads) {
        return 0;
    }

    /**
     * This starts cracking of the specified password length.
     * 
     * @param passwordLength Password length in characters to crack.
     */
    virtual void crackPasswordLength(int passwordLength) = 0;
    
    /**
     * This function is called by the pthread or boost::thread start to start cracking.
     */
    virtual void GPU_Thread() = 0;

protected:
    /**
     * A mutex for changes to the static data structures in MFNHashType.
     */
    boost::mutex MFNHashTypeMutex;


    /**
     * This is set if the hash is of a "Big Endian" type - typically SHA functions.
     */
    static uint8_t HashIsBigEndian;
    
    /**
     * Set up the class for the entry to the multithreaded mode.
     * 
     * This function is called prior to entering the multithreaded mode after
     * all thread IDs have been added.  It is responsible for setting vector
     * sizes and such to the right widths and any other setup required.
     */
    virtual void setupClassForMultithreadedEntry() = 0;

    /**
     * Sets up the device as needed, according to the threadRunDataV passed in.
     * 
     * This function handles setting the thread up for the device as is required
     * by the device environment.  This would include setting the platform and
     * device for OpenCL, setting the device for CUDA, etc.  At the end of this
     * function, all subsequent calls in the thread will reference the specifed
     * device without any additional setup needed.  There may be no setup needed
     * for certain device types (likely CPUs).
     */
    virtual void setupDevice() = 0;
    
    /**
     * Sets up the OpenCL kernels based on the memory allocation state.
     * 
     * This function is mostly useful for OpenCL - this is called AFTER the 
     * memory allocation functions, and is intended to be used for compiling
     * the OpenCL kernel based on the successfully allocated bitmaps/etc.  For
     * other device types, this is probably not very useful.
     */
    virtual void doKernelSetup() { };
    
    /**
     * Tear down any device context, fully closing out the device.
     * 
     * This function tears down whatever setupDevice did, and fully closes out
     * of the device, preparing the thread and the device for future use.  At
     * the end of this function, the device can be reused by setupDevice.
     */
    virtual void teardownDevice() = 0;
    
    /**
     * Allocate GPU and thread-specific memory as needed.
     * 
     * This function will be called to handle allocation of GPU specific memory
     * and host memory related to this GPU memory.  This should handle
     * everything needed by the class for operation.
     */
    virtual void allocateThreadAndDeviceMemory() = 0;
    
    /**
     * Free all GPU and thread-specific memory as needed.
     * 
     * This function frees all the various memory allocated by
     * allocateThreadAndDeviceMemory.  When this function is done, all 
     * memory allocated will be fully freed.
     */
    virtual void freeThreadAndDeviceMemory() = 0;
    
    /**
     * Copies run-specific data to the device.
     * 
     * This function copies all the run-specific data to the device, using
     * the memory allocated in allocateThreadAndDeviceMemory.  This would
     * include the bitmaps, the hash list, etc.  It does NOT copy the constant
     * space data - this needs to be called separately.
     */
    virtual void copyDataToDevice() = 0;
    
    /**
     * Copies a workunit-specific wordlist to the device.
     * 
     * This will copy a chunk of workunit to the device.  For now, it will copy
     * everything, but in the future it may handle chunks to deal with less
     * device memory.
     * 
     * @param wordlistLengths A vector containing the word lengths.
     * @param wordlistData A vector containing the word data.
     */
    virtual void copyWordlistToDevice(std::vector <uint8_t> &wordlistLengths,
        std::vector<uint32_t> &wordlistData) { };
    
    /**
     * This function will be called at the beginning of each workunit, and can
     * be used to perform workunit-specific setup.  This will be used for the
     * salted hash type (to update the salt list on the GPU), and for wordlist
     * types (to load new wordlists).  The default is to do nothing.
     */
    virtual void doPerWorkunitDeviceSetup() {
        
    }


    /**
     * Copies run-specific constant data to the device.
     *
     * This function copies constant data to the device as needed.  The reason
     * this is a separate function is that in order to copy constant data to the
     * GPUs, the memcpy call must come from within the same compilation unit
     * as the constant defined region.  This basically means that there needs
     * to be a C function within the .cu file to handle this.  This function,
     * in the C++ section, will serve as a trampoline into the extern C function
     * defined in the .cu file that actually copies the data.
     */
    virtual void copyConstantDataToDevice() = 0;
    
    /**
     * Pre-processes a hash based on the specific algorithm.
     * 
     * This function is hash-specific and involves pre-processing a hash based
     * on the password length and anything else that can be done to reduce the
     * effort needed to find it.  This is called on each hash from the hash
     * file class before it is copied to the device (and before bitmaps are
     * set up).
     * 
     * Default behavior is to return the unmodified hash.
     * 
     * @param rawHash The raw hash out of the hashfile that should be processed.
     * @return The modified version of the hash that will be passed to the device.
     */
    virtual std::vector<uint8_t> preProcessHash(std::vector<uint8_t> rawHash) {
        return rawHash;
    }
    
    /**
     * Post-processes a hash to convert the pre-processed version back to the raw version.
     * 
     * This function does whatever post-processing is needed to convert the
     * pre-processed hash back into the raw form that came from the hash file.
     * This is called before pushing a hash back to the hash file as a found hash.
     * 
     * Default behavior is to return the unmodified hash.
     * 
     * @param processedHash
     * @return 
     */
    virtual std::vector<uint8_t> postProcessHash(std::vector<uint8_t> processedHash) {
        return processedHash;
    }

    /**
     * Convert the hash vector array into the format needed by the GPU.
     *
     * This function copies the hashes from the vector array into a region
     * of memory that can be used to copy the data to the GPU.  This is done
     * once per thread to avoid massive duplication of memory on heavily
     * multi-GPU systems.
     */
    virtual void copyHashesIntoDeviceFormat() = 0;

    /**
     * Launches the kernel.  Will need other parameters.
     */
    virtual void launchKernel() = 0;
    
    /**
     * Platform-specific method of copying start points to the device.
     */
    virtual void copyStartPointsToDevice() = 0;
    
    /**
     * Create the needed lookup bitmaps based on the hash.
     * 
     * This function creates the lookup bitmaps to push to the device.  However,
     * the bitmaps must be created after the hashes are pre-processed.  This
     * will need to be called after the hashes are processed for the current
     * password length.
     */
    virtual void createLookupBitmaps() = 0;
    
    /**
     * A function to sort the hashes in the class.
     */
    virtual void sortHashes() = 0;

    /**
     * Performs a device-class specific synchronization.
     *
     * This will call whatever the needed device sync function is -
     * cudaThreadSynchronize(), etc.
     */
    virtual void synchronizeThreads() = 0;

    /**
     * Set up the start points for each thread.
     *
     * This function creates the start points as needed for each thread.  They
     * are stored in the appropriate host space for copying to the device.
     *
     * @param perThread How many hashes each thread will check.
     * @param startPoint The start offset in password space to calculate for.
     */
    virtual void setStartPoints(uint64_t perThread, uint64_t startPoint) = 0;

    /**
     * Copies the device found password list to the host.
     *
     * This function copies the device found password list and success flags
     * to the host for analysis.
     *
     * @param threadData The run data for the thread being copied.
     */
    virtual void copyDeviceFoundPasswordsToHost() = 0;

    /**
     * Outputs found hashes to the display and hash class.
     *
     * @param threadData The run data for the thread being calculated.
     */
    virtual void outputFoundHashes() = 0;
    
    /**
     * Gets a list of filenames to use for OpenCL compilation.  Returns the vector
     * of strings.
     * @return A vector of filenames to use for OpenCL complilation.
     */
    virtual std::vector<std::string> getHashFileNames() {
        std::vector<std::string> returnNull;
        return returnNull;
    };
    
    /**
     * Get the kernel OpenCL source as a string.  This is used to support the
     * embedded kernel source.
     * 
     * @return A string containing the OpenCL code to compile.
     */
    virtual std::string getKernelSourceString() {
        std::string returnNull;
        return returnNull;
    }
    
    /**
     * Returns the kernel name to use for OpenCL compilation.
     * @return The kernel name for OpenCL compilation.
     */
    virtual std::string getHashKernelName() {
        std::string returnNull;
        return returnNull;
    }
    
    /**
     * Returns a vector of kernel names for OpenCL compilation.
     * 
     * This is used if a hash type has multiple kernels (perhaps for different
     * lengths or different steps).  Instead of returning one kernel string,
     * a vector of strings is returned.  They are all compiled in order and
     * placed into the kernel vector for use at runtime.
     * 
     * @return A vector of strings of kernel names
     */
    virtual std::vector<std::string> getHashKernelNamesVector() {
        std::vector<std::string> kernelNames;
        return kernelNames;
    }

    /**
     * Get additional OpenCL defines as a string for compilation.
     * @return 
     */
    virtual std::string getDefineStrings() {
        return std::string();
    }
    
    /**
     * Revise the requested thread count if needed to match hardware limits.
     * 
     * Some algorithms may have a high shared memory requirement, which would
     * act to reduce the possible thread count.  This will drop the requested
     * thread count down to what the hardware can support.  This function, if
     * needed, must be aware of the device characteristics.
     * 
     * Default behavior is to return the requested count as-is.
     * 
     * @param requestedThreadCount Number of requested hardware threads.
     * @return Modified number, if lower than the requested count.
     */
    virtual uint32_t getMaxHardwareThreads(uint32_t requestedThreadCount) {
        return requestedThreadCount;
    }


    /**
     * Static pointers to the various classes we will need.  These are constant
     * across all the different classes.
     */
    static MFNCommandLineData *CommandLineData;
    static CHCharsetNew *Charset;
    static MFNWorkunitBase *Workunit;
    static CHHashFileV *HashFile;
    static MFNDisplay *Display;

    /**
     * The current thread ID
     */
    uint32_t threadId;

    /**
     * The number of threads in use.  Incremented each new class added.
     */
    static uint32_t numberThreads;

    /**
     * threadRendezvous is used as a flag to force threads to exit.
     *
     * This flag is set to 1 to force all threads to exit if the password
     * length changes.  This is a work in progress.  It should be 0 for normal
     * operation.
     */
    char threadRendezvous;
};

#endif