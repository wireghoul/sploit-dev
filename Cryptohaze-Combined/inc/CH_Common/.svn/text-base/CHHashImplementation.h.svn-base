/*
Cryptohaze Multiforcer & Wordyforcer - low performance GPU password cracking
Copyright (C) 2012  Bitweasil (http://www.cryptohaze.com/)

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
 *
 * @section DESCRIPTION
 *
 * This set of classes implements basic hash implementations and unwinding for
 * the assorted hash types.  It should handle hashing input strings of any
 * length, and generally be entirely correct for anything thrown at it.
 * 
 * The unwinding should be done in a manner that can be done for all hashes
 * of the specified length, and will be used by the CUDA, OpenCL, and CPU code.
 * 
 * It will be used for password verification as well.  Basically, these have
 * to do everything they are advertised to do fully correctly.
 * 
 * These should NOT be SSE/AVX functions - they are intended to be single hash,
 * and easy to understand.  These are NOT being used for cracking, so don't
 * worry about performance that much!
 * 
 * Please ensure these classes are all threadsafe.  You must assume that
 * multiple threads will be calling these functions, and you should not block.
 * This pretty much means no private variables.
 */

#ifndef __CHHASHIMPLEMENTATION_H__
#define __CHHASHIMPLEMENTATION_H__

#include <string>
#include <vector>
#include <stdint.h>
#include <stddef.h>

class CHHashImplementation {
public:
    /**
     * Perform the specified hash function on the input data.
     * 
     * @param rawData The stream of bytes to hash
     * @return A vector containing the hash value of the sequence of bytes.
     */
    virtual std::vector<uint8_t> hashData(const std::vector<uint8_t> &rawData) 
        = 0;
    
    /**
     * Perform the specified hash and return the data in an ASCII string.
     * 
     * This function replicates the behavior of PHP (and many other language)
     * hash functions that return a lowercase string of data instead of the
     * binary result.  This is useful for web hashes.  The default is to return
     * lowercase letters, as this is what most web languages do, however the
     * user may request uppercase letters instead.
     * 
     * @param rawData The stream of bytes to hash.
     * @param useUppercase True if the letters should be uppercase, else false.
     * @return A vector containing the hash, in ASCII output.  Or a string.
     */
    virtual std::vector<uint8_t> hashDataAsciiVector(
        const std::vector<uint8_t> &rawData,
        uint8_t useUppercase = 0);

    virtual std::string hashDataAsciiString(const std::vector<uint8_t> &rawData,
        uint8_t useUppercase = 0);

    /**
     * Perform the hash operation on multiple input blocks of data.
     * 
     * This function hashes multiple input vectors and returns a vector of hash
     * values corresponding to the hashes of each input vector (in the same
     * order).
     * 
     * @param rawMultipleData A vector of vectors of raw data.
     * @return A vector of the hashes of the raw data.
     */
    virtual std::vector<std::vector<uint8_t> > hashMultipleData(
        const std::vector<std::vector<uint8_t> > &rawMultipleData);

    /**
     * Same as the above ASCII functions, for multiple hashes.
     * 
     * @param rawMultipleData The data to hash.
     * @param useUppercase True if the letters should be uppercase, else false.
     * @return A vector of vectors or strings, ascii.
     */
    virtual std::vector<std::vector<uint8_t> > hashMultipleDataAsciiVector(
        const std::vector<std::vector<uint8_t> > &rawMultipleData,
        uint8_t useUppercase = 0);
    virtual std::vector<std::string> hashMultipleDataAsciiString(
        const std::vector<std::vector<uint8_t> > &rawMultipleData,
        uint8_t useUppercase = 0);
    
    /**
     * Perform the common tasks needed to prepare the hash for use on the GPUs
     * or CPU cracking framework.  This includes:
     * - Reordering the byte order to little endian if needed
     * - Performing unwinding on the hash, if possible, as specified for the
     * password length.
     * 
     * This will be highly hash function specific, but in general the pre-work
     * needed for the various hashes should be identical between kernels.  If
     * the hashes are SHA-type hashes, the ordering should be reversed in each
     * word to match the little endian architecture being used for hash
     * cracking.  If the hash function can be unwound partially (for MD5/NTLM),
     * this should be done as well.  When finished, the hashes should be ready
     * for bitmap processing and being sent to the GPU.
     * 
     * This function modifies the passed in vector in place.
     * 
     * @param passLength Password length in bytes.  If 0, no length
     *  specific processing is performed. 
     * @param rawHash The vector containing the raw hash to process.
     */
    virtual void prepareHash(int passLength, std::vector<uint8_t> &rawHash) = 0;
    
    /**
     * Post-process a hash - this "completes" the hash as generated by
     * prepareHash.
     * 
     * This is basically the reverse of the above function.  When the hash is
     * found on the device, it will need to be converted back to the "proper"
     * hash to report to the HashFile class.  This function does that.
     * 
     * @param passLength Password length in bytes.  If 0, no processing happens.
     * @param rawHash The vector containing the hash to post-process.
     */
    virtual void postProcessHash(int passLength, std::vector<uint8_t> &rawHash)
        = 0;

    /**
     * Same as above, but performs the operation on multiple hashes.
     * 
     * @param passLength Password length in bytes, or 0 to disable unwinding.
     * @param rawMultipleHash A vector of the hashes to process.
     */
    virtual void prepareMultipleHash(int passLength,
        std::vector<std::vector<uint8_t> > &rawMultipleHash);
    
};

#endif
