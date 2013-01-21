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
 * The class MFNHashTypePlain is the plain hash (unsalted) class implementation
 * for the new MFNHashType classes.  This class is still technology independent,
 * and implements the common functionality for hash types.  This handles the
 * basic setup and running of plain, unsalted hashes.
 *
 * This class also deals with the fact that some hashes are big endian and some
 * are little endian.  This refers to the loading of registers as compared to
 * the byte order in the hash.  MD5 is little endian, as you can simply load
 * the hash by converting the uint8_t array to a uint32_t pointer, and loading
 * the registers on a little endian architecture.  SHA1 is big endian, as the
 * hash to registers are loaded as though it is a big endian architecture.
 *
 * This is important to improve search performance.  The goal is for the search
 * algorithm on the device to never have to invert the byte ordering for hash
 * searching.  While this isn't a huge deal, as the bitmaps should take care
 * of most of this, it's still cleaner than the old method where the registers
 * have to be swapped for sorting and searching.
 */

#ifndef __MFNHASHTYPEPLAIN_H
#define __MFNHASHTYPEPLAIN_H

#include "MFN_Common/MFNHashType.h"
#include "MFN_Common/MFNDebugging.h"
#include "CH_Common/CHHiresTimer.h"

/**
 * This structure is used to describe hash attributes that each hash type needs.
 * 
 * By doing this, each hash type can request only capabilities it needs, and the
 * rest is not generated.  This allows the request of specific bitmaps, data
 * regions on the GPU, etc.  In theory, this will let us use resources more
 * efficiently, and allow future extension with new capabilities available to
 * all hash types.  Each chunk of attributes are described individually.
 * 
 * This will be a static structure, so only set the things you need - do not set
 * attributes to 0, as other devices may need them.
 */
typedef struct {
    /**
     * Specify the hash "native word size" - default is 32-bit.
     */
    char hashWordWidth32;
    char hashWordWidth64;
    
    /**
     * Specify the creation of 128MB bitmaps for a-h (words 0-7).  If specified,
     * the bitmap will be created for the given word in the hash.  Note that
     * specifying a word beyond what is valid is meaningless and will be
     * ignored.  These assume 32-bit words right now... probably will need to
     * fix this going forward.
     */
    char create128mbBitmapA;
    char create128mbBitmapB;
    char create128mbBitmapC;
    char create128mbBitmapD;
    char create128mbBitmapE;
    char create128mbBitmapF;
    char create128mbBitmapG;
    char create128mbBitmapH;
    
    /**
     * Create 256kb bitmaps for L1/L2 cache
     */
    char create256kbBitmapA;
    char create256kbBitmapB;
    char create256kbBitmapC;
    char create256kbBitmapD;
    char create256kbBitmapE;
    char create256kbBitmapF;
    char create256kbBitmapG;
    char create256kbBitmapH;
    
    /**
     * Create 32kb, 16kb, 8kb, 4kb bitmaps for shared memory.
     */
    char create32kbBitmapA;
    char create16kbBitmapA;
    char create8kbBitmapA;
    char create4kbBitmapA;
    char create32kbBitmapB;
    char create16kbBitmapB;
    char create8kbBitmapB;
    char create4kbBitmapB;
    char create32kbBitmapC;
    char create16kbBitmapC;
    char create8kbBitmapC;
    char create4kbBitmapC;
    char create32kbBitmapD;
    char create16kbBitmapD;
    char create8kbBitmapD;
    char create4kbBitmapD;
    char create32kbBitmapE;
    char create16kbBitmapE;
    char create8kbBitmapE;
    char create4kbBitmapE;
    char create32kbBitmapF;
    char create16kbBitmapF;
    char create8kbBitmapF;
    char create4kbBitmapF;
    char create32kbBitmapG;
    char create16kbBitmapG;
    char create8kbBitmapG;
    char create4kbBitmapG;
    char create32kbBitmapH;
    char create16kbBitmapH;
    char create8kbBitmapH;
    char create4kbBitmapH;
    
    /**
     * If the hash type is salted, this enables the copying of salt values to
     * the GPU and the reloading of them on workunit updates.
     */
    char hashUsesSalt;
    
    /**
     * If the salt needs padding at the end (0x80), set this.
     */
    char padSalt;
    
    /**
     * If the hash type uses a wordlist, this will enable the copying of the
     * wordlist to the device and will generally enable wordlist based cracking.
     */
    char hashUsesWordlist;
    
    /**
     * If the hash type uses iterations, this will enable copying of them to the
     * GPU.
     */
    char hashUsesIterationCount;
    
    /**
     * If the hash is using other data types (for IKE hashes, collisions, etc),
     * specify these and they will be copied to the GPU after being requested.
     * Each of these will be set up with a length field and the data coalesced
     * for easy loading.
     */
    char hashUsesData1;
    char hashUsesData2;
    char hashUsesData3;
    char hashUsesData4;
    char hashUsesData5;
    
    /**
     * If the hash needs temp space (perhaps for multi-kernel communication or
     * inter-kernel storage), set this, and it can request temp space in the
     * GPU global memory based on the thread count.  This defaults to 0, but is
     * set to the bytes per element needed (each hash - so a kernel using
     * 4-vectors is using 4 of these per thread).
     */
    uint32_t hashTempSpaceBytesPerElement;
    
} MFNHashTypeAttributes;

/**
 * Interval to report total network rate, in seconds.
 */
#define MFN_NETWORK_REPORTING_INTERVAL 10.0

class MFNHashTypePlain : public virtual MFNHashType {
public:
    /**
     * Constructor, providing the hash length in bytes.
     *
     * This is the constructor for all plain hash types.  This takes the length
     * of the hash in bytes, and handles all the setup for this.
     *
     * @param newHashLengthBytes The length of the hash being cracked.
     */
    MFNHashTypePlain(uint16_t newHashLengthBytes);

    /**
     * This is the entry point for password cracking of a given length.
     *
     * This will start the password cracking and launch all the required threads
     * for this password length.
     *
     * @param passwordLength The password length to crack.
     */
    void crackPasswordLength(int passwordLength);

    void GPU_Thread();

    virtual void sortHashes();

protected:
    /**
     * A mutex for changes to the static data structures in MFNHashTypePlain.
     */
    static boost::mutex MFNHashTypePlainMutex;
    
    // Run a brute force workunit
    virtual void RunGPUWorkunitBF(struct MFNWorkunitRobustElement *WU);
    
    // Run a wordlist workunit
    virtual void RunGPUWorkunitWL(struct MFNWorkunitRobustElement *WU);
    

    virtual void createLookupBitmaps();
    
    /**
     * Create a lookup bitmap based on the provided hash list.
     * 
     * This function creates a bitmap based on the provided information.
     * startWord is what offset to start creating the bitmaps from - 0 for the
     * first word, 1 for the second word, etc.
     * The hashlist is passed in by reference, and is the list to create the 
     * bitmap for - either the raw list or the preprocessed list.
     * Finally, the result goes in the bitmap vector.
     * Note that the bitmap uses the LOW order bits of each word, to prevent
     * the GPU from having to do an extra shift - it can just bitmask and do
     * one shift instead of having to shift the whole word.
     * 
     * The 8kb bitmaps are defined as follows:
     * 8192 bytes, 8 bits per byte.
     * Bits set are determined by the low 16 bits of each point in the hash.
     * Byte index is 13 bits, bit index is 3 bits.
     * Algorithm: To set:
     * First 13 bits of the hash (high order bits) are used as the index to the array.
     * Next 3 bits control the left-shift amount of the '1' bit.
     *
     * The startword is which word (32-bits) to create the bitmap for.
     * The hashlist is passed in by reference.
     * The bitmap128mb is the vector to create the bitmap in.
     *
     *
     * @param startWord
     * @param hashList
     * @param bitmap
     * @param bitmapSizeBytes Power of 2 size for bitmap.
     */
    virtual void createArbitraryBitmap(uint8_t startWord,
            std::vector<std::vector<uint8_t> > &hashList, std::vector<uint8_t> &bitmap128mb,
            uint32_t bitmapSizeBytes);

    /**
     * Convert a vector of input data to a set of vectors suited to device access.
     * 
     * This function generically takes a vector of input vectors, and converts them
     * into a vector of data that is 32-bit coalesced for GPU access.  In other
     * words, the first 4 bytes of data0 are followed by the first 4 bytes of data1,
     * etc.  This enables coalesced access when GPUs are accessing data - if threads
     * all fetch word0, they will be accessing a contiguous region of memory.
     * 
     * This also creates a vector of 32-bit values that contain the length of each
     * element, in little endian format.  Easy access for the GPU!  This length does
     * NOT include the length of the data offset, or the padding bit - it is the
     * length of the data only.
     * 
     * Note that this function will OVERWRITE ALL DATA in the two output vectors!
     * 
     * isBigEndianData should be set if the data coming in is for a big endian hash
     * such as the SHA family, otherwise should be false.
     * 
     * addPaddingBit will append a padding bit to the end of each data element
     * before inserting it.  This is useful for salts that come after the password.
     * 
     * dataOffset is used to allow for pass.salt constructs more easily.  If this is
     * set, the data will be offset this many bytes in the first word.  This allows
     * more easily oring the salt in after the password.  This value should be in
     * the range of 0-3 or it makes no sense (or, more specifically, will be %4'd).
     * 
     * @param inputData Input data vector (vector of vectors of uint8_t)
     * @param isBigEndian Set to true for big endian hashes (SHA family, etc)
     * @param addPaddingBit Set to true to append 0x80 to all input data.
     * @param dataOffset Data offset for the first word of data.
     * @param dataLengths Output of the lengths of each element.
     * @param dataDeviceFormat Output of the coalesced data in 32-bit chunks.
     */
    virtual void convertData32(
        const std::vector<std::vector<uint8_t> > &inputData, char isBigEndian,
        char addPaddingBit, uint8_t dataOffset,
        std::vector<uint32_t> &dataLengths,
        std::vector<uint32_t> &dataDeviceFormat);
    
    /**
     * Convert wordlist data to the device format.  This involves altering the
     * data such that all of the word0 data is in contiguous chunks of memory,
     * all of word1 data is the same, etc.  This is optimal for GPU loading.
     * 
     * The data can also be optionally aligned by byte widths for ensuring
     * alignment in GPU memory for each set, which should optimize vector
     * loading and other fun things.
     * 
     * @param inputWordlistLengths Input lengths
     * @param inputWordlistBlocks Input block data
     * @param outputWordlistLengths Output lengths
     * @param outputWordlistBlocks Output block data
     * @param byteAlignOffset Byte alignment (16 or so is good)
     */
    virtual void covertWordlist32(
        std::vector<uint8_t> &inputWordlistLengths,
        std::vector<uint32_t> &inputWordlistBlocks,
        std::vector<uint8_t> &outputWordlistLengths,
        std::vector<uint32_t> &outputWordlistBlocks,
        uint32_t byteAlignOffset);
    
    /**
     * Copies the hashes from their vector of vectors format into a single
     * array that the device uses.
     */
    virtual void copyHashesIntoDeviceFormat();
    
    /**
     * Converts the vector of charset vectors into a single array of the length
     * that the device kernels use.
     */
    virtual void setupCharsetArrays();

    /**
     * Stores the specified start points as the actual characters that will go
     * onto the device, in 32-bit wide intervals, padded with the padding bit.
     *
     * This allows the device to simply load the values from this array, without
     * having to read the charset array.  It can load from the created array
     * into b0/b1/etc, and begin processing immediately.  This will likely be
     * used by all of the GPU and CPU kernels, as they can all load a vector
     * of 32-bit values in a sane parallel fashion.
     */
    virtual void setStartPasswords32(uint64_t perThread, uint64_t startPoint);

    /**
     * This method can be implemented by derived classes to perform additional
     * static data setup (as for salted hashes or other types).
     */
    virtual void doAdditionalStaticDataSetup();
    
    /**
     * Sets the maximum plantext length on each kernel invocation.  This is used
     * to allow kernels to report hashes longer than the plain length if needed
     * for whatever reason (rules, duplicated hashes, etc).  Default behavior
     * is to use the current password length.
     */
    virtual void setMaxFoundPlainLength() {
        this->maxFoundPlainLength = this->passwordLength;
    }

    // ============ Salted hash functions ============
    
   /**
     * Get the list of salts for uncracked hashes from the HashFile class.
     * Populate the salt length list with the actual length of the salt.
     * If requested, add the padding bit at the end.  Set up the device format
     * salt array with little endian or big endian salts as needed.  This
     * function manages its own locks, but data should not be copied out while
     * the locks are locked.
     */
    virtual void setupActiveSaltArrays();
    
    /**
     * Optionally overridable function that returns the offset for the salts
     * in device format.  This is used for (pass.salt) types, and allows the
     * salt to be offset in the vector that the vector can be simply or'd with
     * the password and generate the right value.
     */
    virtual int getSaltOffset() {
        return 0;
    }
    
    /**
     * Optionally overridable function that determines if the padding bit should
     * be set at the end of the salt.  Typically, this will be true for simple
     * (pass.salt) types, and false for others.
     */
    virtual int getSetSaltPaddingBit() {
        return 0;
    }
    
    /**
     * Override the per-workunit setup
     */
    virtual void doPerWorkunitDeviceSetup();
    
    /**
     * Copy the salt arrays to the device.  This will be called from multiple
     * functions, so is a separate function!  Must be implemented in the
     * device specific classes.
     */
    virtual void copySaltArraysToDevice() = 0;
    
    
    /**
     * True if the various data is initialized, else false.  Used when threads
     * acquire the mutex to see if they must do anything or if they can go
     * directly to device setup.
     */
    static uint8_t staticDataInitialized;
   
    /**
     * The length of the hash type, in bytes.  This must be the same for all threads.
     */
    static uint16_t hashLengthBytes;

    /**
     * Current password length being cracked.  Same for all threads.
     */
    static uint16_t passwordLength;
    
    /**
     * Maximum plain length.  This is the maximum plaintext length that can be
     * reported.  This is often the same as passwordLength, but can be different
     * for kernels such as DUPMD5 or things using wordlists.  This is set by the
     * virtual function setMaxFoundPlainLength, which defaults to setting it to
     * the password length (to keep memory requirements low).  This can be 
     * overridden by specific hash types if needed (duplicated hashes, etc).
     */
    static uint16_t maxFoundPlainLength;
    
    /**
     * Number of 32-bit words needed in total for each password.  This is the
     * password length + 1 (for padding) rounded up to the nearest 4 bytes.
     */
    static uint16_t passwordLengthWords;


    /**
     * A copy of the active hashes - this is the raw version, not the modified
     * version.
     */
    static std::vector<std::vector<uint8_t> > activeHashesRaw;

    /**
     * A copy of the pre-processed hashes.
     */
    static std::vector<std::vector<uint8_t> > activeHashesProcessed;

    /**
     * A list of the processed hashes, in the format the device wants - a long
     * list of bytes.  Created by copyHashesIntoDeviceFormat().
     */
    static std::vector<uint8_t> activeHashesProcessedDeviceformat;

    /**
     * The current charset being used.
     */
    static std::vector<std::vector<uint8_t> > currentCharset;

    /**
     * Vectors for 4kb bitmaps (for shared memory).
     */
    static std::vector<uint8_t> sharedBitmap4kb_a;
    static std::vector<uint8_t> sharedBitmap4kb_b;
    static std::vector<uint8_t> sharedBitmap4kb_c;
    static std::vector<uint8_t> sharedBitmap4kb_d;
    static std::vector<uint8_t> sharedBitmap4kb_e;
    static std::vector<uint8_t> sharedBitmap4kb_f;
    static std::vector<uint8_t> sharedBitmap4kb_g;
    static std::vector<uint8_t> sharedBitmap4kb_h;

    /**
     * Vectors for 8kb bitmaps (for shared memory).
     */
    static std::vector<uint8_t> sharedBitmap8kb_a;
    static std::vector<uint8_t> sharedBitmap8kb_b;
    static std::vector<uint8_t> sharedBitmap8kb_c;
    static std::vector<uint8_t> sharedBitmap8kb_d;
    static std::vector<uint8_t> sharedBitmap8kb_e;
    static std::vector<uint8_t> sharedBitmap8kb_f;
    static std::vector<uint8_t> sharedBitmap8kb_g;
    static std::vector<uint8_t> sharedBitmap8kb_h;

    /**
     * Vectors for 16kb bitmaps (for shared memory)
     */
    static std::vector<uint8_t> sharedBitmap16kb_a;
    static std::vector<uint8_t> sharedBitmap16kb_b;
    static std::vector<uint8_t> sharedBitmap16kb_c;
    static std::vector<uint8_t> sharedBitmap16kb_d;
    static std::vector<uint8_t> sharedBitmap16kb_e;
    static std::vector<uint8_t> sharedBitmap16kb_f;
    static std::vector<uint8_t> sharedBitmap16kb_g;
    static std::vector<uint8_t> sharedBitmap16kb_h;

    /**
     * Vectors for 32kb bitmaps (for shared memory)
     */
    static std::vector<uint8_t> sharedBitmap32kb_a;
    static std::vector<uint8_t> sharedBitmap32kb_b;
    static std::vector<uint8_t> sharedBitmap32kb_c;
    static std::vector<uint8_t> sharedBitmap32kb_d;
    static std::vector<uint8_t> sharedBitmap32kb_e;
    static std::vector<uint8_t> sharedBitmap32kb_f;
    static std::vector<uint8_t> sharedBitmap32kb_g;
    static std::vector<uint8_t> sharedBitmap32kb_h;

    /**
     * Vectors for the 256kb global memory small bitmaps
     * These should fit nicely in L2 cache.
     */
    static std::vector<uint8_t> globalBitmap256kb_a;
    static std::vector<uint8_t> globalBitmap256kb_b;
    static std::vector<uint8_t> globalBitmap256kb_c;
    static std::vector<uint8_t> globalBitmap256kb_d;
    static std::vector<uint8_t> globalBitmap256kb_e;
    static std::vector<uint8_t> globalBitmap256kb_f;
    static std::vector<uint8_t> globalBitmap256kb_g;
    static std::vector<uint8_t> globalBitmap256kb_h;

    /**
     * Vectors for the 128MB bitmaps (for global memory)
     */
    static std::vector<uint8_t> globalBitmap128mb_a;
    static std::vector<uint8_t> globalBitmap128mb_b;
    static std::vector<uint8_t> globalBitmap128mb_c;
    static std::vector<uint8_t> globalBitmap128mb_d;
    static std::vector<uint8_t> globalBitmap128mb_e;
    static std::vector<uint8_t> globalBitmap128mb_f;
    static std::vector<uint8_t> globalBitmap128mb_g;
    static std::vector<uint8_t> globalBitmap128mb_h;
    
    /**
     * The charsetForwardLookup is a forward lookup vector for the charset space.
     * It is passwordLen * 128 bytes long, and contains at each 128 byte
     * boundary the characters for the next position.  If the charset is a
     * single charset, the length is only 128 bytes.
     */
    static std::vector<uint8_t> charsetForwardLookup;
    
    /**
     * The charsetReverseLookup is an experiment to reduce the kernel
     * register count and try for improved performance.  This maps the value
     * of each character to the position in the charset.
     */

    static std::vector<uint8_t> charsetReverseLookup;
    
    /**
     * charsetLengths contains the length of the charset for each
     * password position.  This is used to determine when to wrap.
     */
    static std::vector<uint8_t> charsetLengths;


    /**
     * Number of steps to run per thread.  Persists across workunits.  Per-thread.
     */
    uint32_t perStep;
    
    /**
     * Step to start on, for wordlist use.
     */
    uint32_t startStep;

    /**
     * True if a single charset is used, false if a multi charset is used.
     */
    static uint8_t isSingleCharset;

    /**
     * The client ID from the workunit class to handle per-thread cancellation.
     */
    uint16_t ClientId;

    /**
     * Number of threads to run.  On CPU tasks, this will be the number of threads.
     */
    uint32_t GPUThreads;

    /**
     * Number of blocks to run.  On CPU tasks, this will be 1.
     */
    uint32_t GPUBlocks;
    
    /**
     * Vector width: How many vectors wide each thread is.
     */
    uint32_t VectorWidth;

    /**
     * The total width of the kernel - how many hashes in parallel.
     * GPUBlocks * GPUThreads * VectorWidth
     */
    uint32_t TotalKernelWidth;
    
    /**
     * Target kernel execution time in ms.  0 for "run until done."
     */
    uint32_t kernelTimeMs;

    /**
     * Target GPU device ID, if relevant.
     */
    uint16_t gpuDeviceId;

    /**
     * Target OpenCL platform ID, if relevant.
     */
    uint16_t openCLPlatformId;

    /**
     * Vector containing the start values as 4-byte values, ready to be loaded
     * directly into b0/b1/etc, with the end padding bit set.  These are loaded
     * with the actual characters to start with, and do not need to be further
     * processed.  This is packed in a sane format for GPUs.  This means that it
     * stores all the b0s together, then all the b1s together, and so forth.
     */
    std::vector<uint8_t> HostStartPasswords32;
    
    
    /**
     * This is the number of unique salts present in the current algorithm.
     * Knowing this allows for the number of steps to be specified with the
     * salt start offset for salted hashes, and gives finer control over the
     * number of steps run (it does not have to be a multiple of the number of
     * unique salts).  This is very useful for large numbers of salts, and for
     * longer running salted algorithms (the shacrypt series, etc).  For
     * unsalted hashes, this is always forced to 1.
     */
    uint32_t numberUniqueSalts;
    
    /**
     * Which salt to start with.  Passed to the device for salted hashes.
     */
    uint32_t saltStartOffset;
    
    /**
     * Iteration offset to start with.  Passed to the device for iterated
     * hashes.
     */
    uint32_t iterationStartOffset;
    
    /**
     * This is used for tracking the time since the rate was last reported
     * over the network.  It will send the rate at the specified interval.
     */
    static CHHiresTimer NetworkSpeedReportingTimer;
    
    /**
     * Currently requested hash attributes - set by each hash leaf node.
     */
    MFNHashTypeAttributes hashAttributes;
    
    /**
     * For the active wordlist, the length of the current wordlist entries
     * in blocks.  This is used for choosing the active kernel to run.
     */
    uint16_t wordlistBlockLength;
    
    // =============== Salted stuff ================
    
    /**
     * True if the various data is initialized, else false.  Used when threads
     * acquire the mutex to see if they must do anything or if they can go
     * directly to device setup.
     */
    static uint8_t saltedStaticDataInitialized;

    /**
     * A mutex for changes to the static data structures in MFNHashTypeSalted.
     */
    static boost::mutex MFNHashTypeSaltedMutex;

    /**
     * This contains a vector of salt lengths in bytes.  It's a 32-bit word for
     * two reasons - insane salt length support, and because most of the
     * supported architectures have a 32-bit word length at some point.
     * 
     * This MUST match the order of salts in the other salt arrays!
     */
    static std::vector<uint32_t> saltLengths;
    
    /**
     * This vector contains all the active salts in vector form (as is exported
     * from the hashfile class).  Whenever this is updated, saltLengths should
     * be updated to match the corresponding order.
     */
    static std::vector<std::vector<uint8_t> > activeSalts;

    /**
     * This vector contains the active salts in device format, which looks like
     * all the word 0 together, all the word 1 together, etc.  Right now, this
     * packs things "tightly" based on the number of unique salts, but this may
     * be altered to improve coalescing if needed.  These values may optionally
     * have the padding bit set.
     */
    static std::vector<uint32_t> activeSaltsDeviceformat;
    
    /**
     * This vector contains all the iteration counts for the corresponding salts
     * above.  Iteration count 0 corresponds to salt 0, etc.  This is used for
     * iterated hashes such as phpbb.
     */
    static std::vector<uint32_t> activeIterationCounts;
    
    /**
     * Other data holders for salted data.  This is used for some of the weirder
     * hash types like IKE.
     */
    static std::vector<std::vector<uint8_t> > otherData1;
    static std::vector<uint32_t> otherData1Deviceformat;
    static std::vector<std::vector<uint8_t> > otherData2;
    static std::vector<uint32_t> otherData2Deviceformat;
    static std::vector<std::vector<uint8_t> > otherData3;
    static std::vector<uint32_t> otherData3Deviceformat;
    static std::vector<std::vector<uint8_t> > otherData4;
    static std::vector<uint32_t> otherData4Deviceformat;
    static std::vector<std::vector<uint8_t> > otherData5;
    static std::vector<uint32_t> otherData5Deviceformat;
    

};

#endif