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
 * This file implements common CUDA cracking defines & functionality.
 */

#if !defined(__CUDACC__)
    // define the keywords, so that the IDE does not complain about them
    #define __global__
    #define __device__
    #define __shared__
    #define __constant__
    #define blockIdx.x 1
    #define blockDim.x 1
    #define threadIdx.x 1
    #define __align__(16)
#endif

#ifndef __MFN_CUDA_COMMON_H__
#define __MFN_CUDA_COMMON_H__

#include "MFN_Common/MFNDefines.h"

/**
 * thread_index is each thread's unique ID in the kernel space.  This is used
 * throughout the kernels, and will be expanded when needed.
 */
#define thread_index (blockIdx.x * blockDim.x + threadIdx.x)

/**
 * BREGS expands out to a comma separated list of all the b0-b15 registers.
 * Conveniently, this is the parameter list for ResetCharacterAtPosition
 */
#define BREGS b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15


/**
 * This is a macro that expands out to load the password from the calculated start
 * points.  This has ideally been superceeded by the functionality of
 * loadPasswords32.  However, this is available if needed.
 *
 * @param sc The sharedCharset name.  Should be in shared mem.
 * @param gsp The globalStartPositions name.
 * @param dt The deviceThreads count name.
 * @param pl The password length.
 */
#define loadPasswordSingle(sc, gsp, dt, pl) { \
if (pl > 0) {ResetCharacterAtPosition(sc[gsp[0 * dt + thread_index]], 0, BREGS);} \
if (pl > 1) {ResetCharacterAtPosition(sc[gsp[1 * dt + thread_index]], 1, BREGS);} \
if (pl > 2) {ResetCharacterAtPosition(sc[gsp[2 * dt + thread_index]], 2, BREGS);} \
if (pl > 3) {ResetCharacterAtPosition(sc[gsp[3 * dt + thread_index]], 3, BREGS);} \
if (pl > 4) {ResetCharacterAtPosition(sc[gsp[4 * dt + thread_index]], 4, BREGS);} \
if (pl > 5) {ResetCharacterAtPosition(sc[gsp[5 * dt + thread_index]], 5, BREGS);} \
if (pl > 6) {ResetCharacterAtPosition(sc[gsp[6 * dt + thread_index]], 6, BREGS);} \
if (pl > 7) {ResetCharacterAtPosition(sc[gsp[7 * dt + thread_index]], 7, BREGS);} \
if (pl > 8) {ResetCharacterAtPosition(sc[gsp[8 * dt + thread_index]], 8, BREGS);} \
if (pl > 9) {ResetCharacterAtPosition(sc[gsp[9 * dt + thread_index]], 9, BREGS);} \
if (pl > 10) {ResetCharacterAtPosition(sc[gsp[10 * dt + thread_index]], 10, BREGS);} \
if (pl > 11) {ResetCharacterAtPosition(sc[gsp[11 * dt + thread_index]], 11, BREGS);} \
if (pl > 12) {ResetCharacterAtPosition(sc[gsp[12 * dt + thread_index]], 12, BREGS);} \
if (pl > 13) {ResetCharacterAtPosition(sc[gsp[13 * dt + thread_index]], 13, BREGS);} \
if (pl > 14) {ResetCharacterAtPosition(sc[gsp[14 * dt + thread_index]], 14, BREGS);} \
if (pl > 15) {ResetCharacterAtPosition(sc[gsp[15 * dt + thread_index]], 15, BREGS);} \
if (pl > 16) {ResetCharacterAtPosition(sc[gsp[16 * dt + thread_index]], 16, BREGS);} \
}

/**
 * This is a macro that expands out to load the password from the calculated start
 * points.  This has ideally been superceeded by the functionality of
 * loadPasswords32.  However, this is available if needed.
 *
 * @param sc The sharedCharset name.  Should be in shared mem.
 * @param gsp The globalStartPositions name.
 * @param dt The deviceThreads count name.
 * @param pl The password length.
 * @param csl The maximum charset length
 */
#define loadPasswordMultiple(sc, gsp, dt, pl, csl) { \
if (pl > 0) {ResetCharacterAtPosition(sc[(csl * 0) + gsp[0 * dt + thread_index]], 0, BREGS);} \
if (pl > 1) {ResetCharacterAtPosition(sc[(csl * 1) + gsp[1 * dt + thread_index]], 1, BREGS);} \
if (pl > 2) {ResetCharacterAtPosition(sc[(csl * 2) + gsp[2 * dt + thread_index]], 2, BREGS);} \
if (pl > 3) {ResetCharacterAtPosition(sc[(csl * 3) + gsp[3 * dt + thread_index]], 3, BREGS);} \
if (pl > 4) {ResetCharacterAtPosition(sc[(csl * 4) + gsp[4 * dt + thread_index]], 4, BREGS);} \
if (pl > 5) {ResetCharacterAtPosition(sc[(csl * 5) + gsp[5 * dt + thread_index]], 5, BREGS);} \
if (pl > 6) {ResetCharacterAtPosition(sc[(csl * 6) + gsp[6 * dt + thread_index]], 6, BREGS);} \
if (pl > 7) {ResetCharacterAtPosition(sc[(csl * 7) + gsp[7 * dt + thread_index]], 7, BREGS);} \
if (pl > 8) {ResetCharacterAtPosition(sc[(csl * 8) + gsp[8 * dt + thread_index]], 8, BREGS);} \
if (pl > 9) {ResetCharacterAtPosition(sc[(csl * 8) + gsp[9 * dt + thread_index]], 9, BREGS);} \
if (pl > 10) {ResetCharacterAtPosition(sc[(csl * 8) + gsp[10 * dt + thread_index]], 10, BREGS);} \
if (pl > 11) {ResetCharacterAtPosition(sc[(csl * 8) + gsp[11 * dt + thread_index]], 11, BREGS);} \
if (pl > 12) {ResetCharacterAtPosition(sc[(csl * 8) + gsp[12 * dt + thread_index]], 12, BREGS);} \
if (pl > 13) {ResetCharacterAtPosition(sc[(csl * 8) + gsp[13 * dt + thread_index]], 13, BREGS);} \
if (pl > 14) {ResetCharacterAtPosition(sc[(csl * 8) + gsp[14 * dt + thread_index]], 14, BREGS);} \
if (pl > 15) {ResetCharacterAtPosition(sc[(csl * 8) + gsp[15 * dt + thread_index]], 15, BREGS);} \
if (pl > 16) {ResetCharacterAtPosition(sc[(csl * 8) + gsp[16 * dt + thread_index]], 16, BREGS);} \
}


/**
 * The loadPassword32 and storePassword32 methods are the preferred method for loading plains.
 * 
 * These work by loading the b0,b1,b2, etc directly from the memory space
 * as plaintext passwords.  At the end of each kernel execution, the current
 * passwords are stored back to the array.  This prevents the need to transfer
 * more plain start points to each thread when the kernel starts again.
 * 
 * @param pa Password initial array
 * @param dt Device number threads
 * @param pl Password length
 */
#define loadPasswords32(pa, dt, pl) { \
a = thread_index; \
b0 = pa[a]; \
if (pl > 3) {a += dt; b1 = pa[a];} \
if (pl > 7) {a += dt; b2 = pa[a];} \
if (pl > 11) {a += dt; b3 = pa[a];} \
if (pl > 15) {a += dt; b4 = pa[a];} \
if (pl > 19) {a += dt; b5 = pa[a];} \
if (pl > 23) {a += dt; b6 = pa[a];} \
if (pl > 27) {a += dt; b7 = pa[a];} \
if (pl > 31) {a += dt; b8 = pa[a];} \
if (pl > 35) {a += dt; b9 = pa[a];} \
if (pl > 39) {a += dt; b10 = pa[a];} \
if (pl > 43) {a += dt; b11 = pa[a];} \
if (pl > 47) {a += dt; b12 = pa[a];} \
if (pl > 51) {a += dt; b13 = pa[a];} \
}

#define storePasswords32(pa, dt, pl) { \
pa[thread_index + 0] = b0; \
if (pl > 3) {pa[thread_index + (dt * 1)] = b1;} \
if (pl > 7) {pa[thread_index + (dt * 2)] = b2;} \
if (pl > 11) {pa[thread_index + (dt * 3)] = b3;} \
if (pl > 15) {pa[thread_index + (dt * 4)] = b4;} \
if (pl > 19) {pa[thread_index + (dt * 5)] = b5;} \
if (pl > 23) {pa[thread_index + (dt * 6)] = b6;} \
if (pl > 27) {pa[thread_index + (dt * 7)] = b7;} \
if (pl > 31) {pa[thread_index + (dt * 8)] = b8;} \
if (pl > 35) {pa[thread_index + (dt * 9)] = b9;} \
if (pl > 39) {pa[thread_index + (dt * 10)] = b10;} \
if (pl > 43) {pa[thread_index + (dt * 11)] = b11;} \
if (pl > 47) {pa[thread_index + (dt * 12)] = b12;} \
if (pl > 51) {pa[thread_index + (dt * 13)] = b13;} \
}

/**
 * Resets a character at the specified position in a MD5/MD5 little-endian style hash.
 *
 * This function is used to reset a character.  It does not affect any other data
 * present in the b0-b13 registers.  It can be used to set any position.
 *
 * @param character The character value to set at the specified position
 * @param position The position (0-55) to reset the character at.
 * @param b0-b13 References to the b0-b13 registers in use.
 */
__device__ inline void ResetCharacterAtPosition(unsigned char character, unsigned char position,
        uint32_t &b0, uint32_t &b1, uint32_t &b2, uint32_t &b3, uint32_t &b4, uint32_t &b5, uint32_t &b6, uint32_t &b7,
	uint32_t &b8, uint32_t &b9, uint32_t &b10, uint32_t &b11, uint32_t &b12, uint32_t &b13, uint32_t &b14, uint32_t &b15) {

    int offset = position / 4;

    if (offset == 0) {
        b0 &= ~(0x000000ff << (8 * (position % 4)));
        b0 |= character << (8 * (position % 4));
    } else if (offset == 1) {
        b1 &= ~(0x000000ff << (8 * (position % 4)));
        b1 |= character << (8 * (position % 4));
    } else if (offset == 2) {
        b2 &= ~(0x000000ff << (8 * (position % 4)));
        b2 |= character << (8 * (position % 4));
    } else if (offset == 3) {
        b3 &= ~(0x000000ff << (8 * (position % 4)));
        b3 |= character << (8 * (position % 4));
    } else if (offset == 4) {
        b4 &= ~(0x000000ff << (8 * (position % 4)));
        b4 |= character << (8 * (position % 4));
    } else if (offset == 5) {
        b5 &= ~(0x000000ff << (8 * (position % 4)));
        b5 |= character << (8 * (position % 4));
    } else if (offset == 6) {
        b6 &= ~(0x000000ff << (8 * (position % 4)));
        b6 |= character << (8 * (position % 4));
    } else if (offset == 7) {
        b7 &= ~(0x000000ff << (8 * (position % 4)));
        b7 |= character << (8 * (position % 4));
    } else if (offset == 8) {
        b8 &= ~(0x000000ff << (8 * (position % 4)));
        b8 |= character << (8 * (position % 4));
    } else if (offset == 9) {
        b9 &= ~(0x000000ff << (8 * (position % 4)));
        b9 |= character << (8 * (position % 4));
    } else if (offset == 10) {
        b10 &= ~(0x000000ff << (8 * (position % 4)));
        b10 |= character << (8 * (position % 4));
    } else if (offset == 11) {
        b11 &= ~(0x000000ff << (8 * (position % 4)));
        b11 |= character << (8 * (position % 4));
    } else if (offset == 12) {
        b12 &= ~(0x000000ff << (8 * (position % 4)));
        b12 |= character << (8 * (position % 4));
    } else if (offset == 13) {
        b13 &= ~(0x000000ff << (8 * (position % 4)));
        b13 |= character << (8 * (position % 4));
    }
}

/**
 * Searches for a 128 bit little endian (MD5, MD4, etc) hash in the global memory.
 *
 * This function takes the calculated hash values (a, b, c, d), the password
 * in b0, b1, etc (as MD5 style - not NTLM!), and the various global memory pointers
 * and searches for the hash.  If it is found, it reports it in the appropriate
 * method.
 *
 * @param a,b,c,d The calculated hash values.
 * @param b0,b1,b2,b3 The registers containing the input block in MD5 format - not NTLM unicode.
 * @param sharedBitmapA The address of the 8kb bitmap ideally in shared memory
 * @param deviceGlobalBitmap{A,B,C,D} The addresses (or null) of the device global bitmaps.
 * @param deviceGlobalFoundPasswords The address of the found-password array
 * @param deviceGlobalFoundPasswordFlags The address of the found-password flag array
 * @param deviceGlobalHashlistAddress The address of the 128-bit hash global hashlist
 * @param numberOfHashes The number of hashes being searched for currently
 * @param passwordLength The current password length
 * @param algorithmType The algorithm currently being used (from MFNDefines.h)
 */
__device__ inline void checkHash128LE(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d,
        uint32_t &b0, uint32_t &b1, uint32_t &b2, uint32_t &b3, uint8_t *sharedBitmapA,
        uint8_t *deviceGlobalBitmapA, uint8_t *deviceGlobalBitmapB,
        uint8_t *deviceGlobalBitmapC, uint8_t *deviceGlobalBitmapD,
        uint8_t *deviceGlobalFoundPasswords, uint8_t *deviceGlobalFoundPasswordFlags,
        uint8_t *deviceGlobalHashlistAddress, uint64_t numberOfHashes,
        uint8_t passwordLength, uint8_t algorithmType) {
    if ((sharedBitmapA[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) {
        if (!(deviceGlobalBitmapA) || ((deviceGlobalBitmapA[(a >> 3) & 0x07FFFFFF] >> (a & 0x7)) & 0x1)) {
            if (!deviceGlobalBitmapB || ((deviceGlobalBitmapB[(b >> 3) & 0x07FFFFFF] >> (b & 0x7)) & 0x1)) {
                if (!deviceGlobalBitmapC || ((deviceGlobalBitmapC[(c >> 3) & 0x07FFFFFF] >> (c & 0x7)) & 0x1)) {
                    if (!deviceGlobalBitmapD || ((deviceGlobalBitmapD[(d >> 3) & 0x07FFFFFF] >> (d & 0x7)) & 0x1)) {
                        uint32_t search_high, search_low, search_index, current_hash_value;
                        uint32_t *DEVICE_Hashes_32 = (uint32_t *) deviceGlobalHashlistAddress;
                        search_high = numberOfHashes;
                        search_low = 0;
                        while (search_low < search_high) {
                            // Midpoint between search_high and search_low
                            search_index = search_low + (search_high - search_low) / 2;
                            current_hash_value = DEVICE_Hashes_32[4 * search_index];
                            // Adjust search_high & search_low to work through space
                            if (current_hash_value < a) {
                                search_low = search_index + 1;
                            } else {
                                search_high = search_index;
                            }
                            if ((a == current_hash_value) && (search_low < numberOfHashes)) {
                                // Break out of the search loop - search_index is on a value
                                break;
                            }
                        }
                        // Broke out of the while loop

                        // If the loaded value does not match, there are no matches - just return.
                        if (a != current_hash_value) {
                            return;
                        }
                        // We've broken out of the loop, search_index should be on a matching value
                        // Loop while the search index is the same - linear search through this to find all possible
                        // matching passwords.
                        // We first need to move backwards to the beginning, as we may be in the middle of a set of matching hashes.
                        // If we are index 0, do NOT subtract, as we will wrap and this goes poorly.

                        while (search_index && (a == DEVICE_Hashes_32[(search_index - 1) * 4])) {
                            search_index--;
                        }
                        while ((a == DEVICE_Hashes_32[search_index * 4])) {
                            if (b == DEVICE_Hashes_32[search_index * 4 + 1]) {
                                if (c == DEVICE_Hashes_32[search_index * 4 + 2]) {
                                    if (d == DEVICE_Hashes_32[search_index * 4 + 3]) {
                                        // Copy the password to the correct location.
                                        switch (passwordLength) {
                                            case 16:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 15] = (b3 >> 24) & 0xff;
                                            case 15:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 14] = (b3 >> 16) & 0xff;
                                            case 14:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 13] = (b3 >> 8) & 0xff;
                                            case 13:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 12] = (b3 >> 0) & 0xff;
                                            case 12:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 11] = (b2 >> 24) & 0xff;
                                            case 11:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 10] = (b2 >> 16) & 0xff;
                                            case 10:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 9] = (b2 >> 8) & 0xff;
                                            case 9:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 8] = (b2 >> 0) & 0xff;
                                            case 8:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 7] = (b1 >> 24) & 0xff;
                                            case 7:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 6] = (b1 >> 16) & 0xff;
                                            case 6:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 5] = (b1 >> 8) & 0xff;
                                            case 5:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 4] = (b1 >> 0) & 0xff;
                                            case 4:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 3] = (b0 >> 24) & 0xff;
                                            case 3:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 2] = (b0 >> 16) & 0xff;
                                            case 2:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 1] = (b0 >> 8) & 0xff;
                                            case 1:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 0] = (b0 >> 0) & 0xff;
                                        }
                                        deviceGlobalFoundPasswordFlags[search_index] = (unsigned char) algorithmType;
                                    }
                                }
                            }
                            search_index++;
                        }
                    }
                }
            }
        }
    }
}



/**
 * Searches for a 128 bit little endian (MD5, MD4, etc) hash in the global memory.
 *
 * This function takes the calculated hash values (a, b, c, d), the password
 * in b0, b1, etc (as MD5 style - not NTLM!), and the hashlist pointers
 * and searches for the hash.  If it is found, it reports it in the appropriate
 * method.  Note that this is inefficient, and should only be called once the
 * bitmaps have been hit!
 *
 * @param a,b,c,d The calculated hash values.
 * @param b0,b1,b2,b3 The registers containing the input block in MD5 format - not NTLM unicode.
 * @param deviceGlobalFoundPasswords The address of the found-password array
 * @param deviceGlobalFoundPasswordFlags The address of the found-password flag array
 * @param deviceGlobalHashlistAddress The address of the 128-bit hash global hashlist
 * @param numberOfHashes The number of hashes being searched for currently
 * @param passwordLength The current password length
 * @param algorithmType The algorithm currently being used (from MFNDefines.h)
 */
__device__ inline void checkHashList128LE(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d,
        uint32_t &b0, uint32_t &b1, uint32_t &b2, uint32_t &b3, 
        uint32_t &b4, uint32_t &b5, uint32_t &b6, uint32_t &b7,
        uint32_t &b8, uint32_t &b9, uint32_t &b10, uint32_t &b11,
        uint32_t &b12, uint32_t &b13,
        uint8_t *deviceGlobalFoundPasswords, uint8_t *deviceGlobalFoundPasswordFlags,
        uint8_t *deviceGlobalHashlistAddress, uint64_t numberOfHashes,
        uint8_t passwordLength, uint8_t algorithmType) {
    uint32_t search_high, search_low, search_index, current_hash_value;
    uint32_t *DEVICE_Hashes_32 = (uint32_t *) deviceGlobalHashlistAddress;
    search_high = numberOfHashes;
    search_low = 0;
    while (search_low < search_high) {
        // Midpoint between search_high and search_low
        search_index = search_low + (search_high - search_low) / 2;
        current_hash_value = DEVICE_Hashes_32[4 * search_index];
        // Adjust search_high & search_low to work through space
        if (current_hash_value < a) {
            search_low = search_index + 1;
        } else {
            search_high = search_index;
        }
        if ((a == current_hash_value) && (search_low < numberOfHashes)) {
            // Break out of the search loop - search_index is on a value
            break;
        }
    }
    // Broke out of the while loop

    // If the loaded value does not match, there are no matches - just return.
    if (a != current_hash_value) {
        return;
    }
    // We've broken out of the loop, search_index should be on a matching value
    // Loop while the search index is the same - linear search through this to find all possible
    // matching passwords.
    // We first need to move backwards to the beginning, as we may be in the middle of a set of matching hashes.
    // If we are index 0, do NOT subtract, as we will wrap and this goes poorly.

    while (search_index && (a == DEVICE_Hashes_32[(search_index - 1) * 4])) {
        search_index--;
    }
    while ((a == DEVICE_Hashes_32[search_index * 4])) {
        if (b == DEVICE_Hashes_32[search_index * 4 + 1]) {
            if (c == DEVICE_Hashes_32[search_index * 4 + 2]) {
                if (d == DEVICE_Hashes_32[search_index * 4 + 3]) {
                    // Copy the password to the correct location.
                    switch (passwordLength) {
                        case 55:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 54] = (b13 >> 16) & 0xff;
                        case 54:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 53] = (b13 >> 8) & 0xff;
                        case 53:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 52] = (b13 >> 0) & 0xff;
                        case 52:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 51] = (b12 >> 24) & 0xff;
                        case 51:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 50] = (b12 >> 16) & 0xff;
                        case 50:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 49] = (b12 >> 8) & 0xff;
                        case 49:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 48] = (b12 >> 0) & 0xff;
                        case 48:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 47] = (b11 >> 24) & 0xff;
                        case 47:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 46] = (b11 >> 16) & 0xff;
                        case 46:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 45] = (b11 >> 8) & 0xff;
                        case 45:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 44] = (b11 >> 0) & 0xff;
                        case 44:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 43] = (b10 >> 24) & 0xff;
                        case 43:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 42] = (b10 >> 16) & 0xff;
                        case 42:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 41] = (b10 >> 8) & 0xff;
                        case 41:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 40] = (b10 >> 0) & 0xff;
                        case 40:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 39] = (b9 >> 24) & 0xff;
                        case 39:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 38] = (b9 >> 16) & 0xff;
                        case 38:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 37] = (b9 >> 8) & 0xff;
                        case 37:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 36] = (b9 >> 0) & 0xff;
                        case 36:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 35] = (b8 >> 24) & 0xff;
                        case 35:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 34] = (b8 >> 16) & 0xff;
                        case 34:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 33] = (b8 >> 8) & 0xff;
                        case 33:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 32] = (b8 >> 0) & 0xff;
                        case 32:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 31] = (b7 >> 24) & 0xff;
                        case 31:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 30] = (b7 >> 16) & 0xff;
                        case 30:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 29] = (b7 >> 8) & 0xff;
                        case 29:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 28] = (b7 >> 0) & 0xff;
                        case 28:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 27] = (b6 >> 24) & 0xff;
                        case 27:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 26] = (b6 >> 16) & 0xff;
                        case 26:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 25] = (b6 >> 8) & 0xff;
                        case 25:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 24] = (b6 >> 0) & 0xff;
                        case 24:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 23] = (b5 >> 24) & 0xff;
                        case 23:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 22] = (b5 >> 16) & 0xff;
                        case 22:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 21] = (b5 >> 8) & 0xff;
                        case 21:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 20] = (b5 >> 0) & 0xff;
                        case 20:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 19] = (b4 >> 24) & 0xff;
                        case 19:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 18] = (b4 >> 16) & 0xff;
                        case 18:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 17] = (b4 >> 8) & 0xff;
                        case 17:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 16] = (b4 >> 0) & 0xff;
                        case 16:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 15] = (b3 >> 24) & 0xff;
                        case 15:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 14] = (b3 >> 16) & 0xff;
                        case 14:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 13] = (b3 >> 8) & 0xff;
                        case 13:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 12] = (b3 >> 0) & 0xff;
                        case 12:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 11] = (b2 >> 24) & 0xff;
                        case 11:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 10] = (b2 >> 16) & 0xff;
                        case 10:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 9] = (b2 >> 8) & 0xff;
                        case 9:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 8] = (b2 >> 0) & 0xff;
                        case 8:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 7] = (b1 >> 24) & 0xff;
                        case 7:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 6] = (b1 >> 16) & 0xff;
                        case 6:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 5] = (b1 >> 8) & 0xff;
                        case 5:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 4] = (b1 >> 0) & 0xff;
                        case 4:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 3] = (b0 >> 24) & 0xff;
                        case 3:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 2] = (b0 >> 16) & 0xff;
                        case 2:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 1] = (b0 >> 8) & 0xff;
                        case 1:
                            deviceGlobalFoundPasswords[search_index * passwordLength + 0] = (b0 >> 0) & 0xff;
                    }
                    deviceGlobalFoundPasswordFlags[search_index] = (uint8_t) algorithmType;
                }
            }
        }
        search_index++;
    }
}

/**
 * Loads a 16-byte hash (MD5, mostly) as a little endian string
 * 
 * This function converts a 16 byte hash into a 32-byte long lowercase
 * ASCII string for use in double/triple MD5 algorithms.
 */

#define LoadHash16AsLEString(hashLookup) { \
    b1 = (uint32_t)hashLookup[(a >> 16) & 0xff][0] | (uint32_t)hashLookup[(a >> 16) & 0xff][1] << 8 | \
            (uint32_t)hashLookup[(a >> 24) & 0xff][0] << 16 | (uint32_t)hashLookup[(a >> 24) & 0xff][1] << 24; \
    b0 = (uint32_t)hashLookup[(a >> 0) & 0xff][0] | (uint32_t)hashLookup[(a >> 0) & 0xff][1] << 8 | \
            (uint32_t)hashLookup[(a >> 8) & 0xff][0] << 16 | (uint32_t)hashLookup[(a >> 8) & 0xff][1] << 24; \
    b3 = (uint32_t)hashLookup[(b >> 16) & 0xff][0] | (uint32_t)hashLookup[(b >> 16) & 0xff][1] << 8 | \
            (uint32_t)hashLookup[(b >> 24) & 0xff][0] << 16 | (uint32_t)hashLookup[(b >> 24) & 0xff][1] << 24; \
    b2 = (uint32_t)hashLookup[(b >> 0) & 0xff][0] | (uint32_t)hashLookup[(b >> 0) & 0xff][1] << 8 | \
            (uint32_t)hashLookup[(b >> 8) & 0xff][0] << 16 | (uint32_t)hashLookup[(b >> 8) & 0xff][1] << 24; \
    b5 = (uint32_t)hashLookup[(c >> 16) & 0xff][0] | (uint32_t)hashLookup[(c >> 16) & 0xff][1] << 8 | \
            (uint32_t)hashLookup[(c >> 24) & 0xff][0] << 16 | (uint32_t)hashLookup[(c >> 24) & 0xff][1] << 24; \
    b4 = (uint32_t)hashLookup[(c >> 0) & 0xff][0] | (uint32_t)hashLookup[(c >> 0) & 0xff][1] << 8 | \
            (uint32_t)hashLookup[(c >> 8) & 0xff][0] << 16 | (uint32_t)hashLookup[(c >> 8) & 0xff][1] << 24; \
    b7 = (uint32_t)hashLookup[(d >> 16) & 0xff][0] | (uint32_t)hashLookup[(d >> 16) & 0xff][1] << 8 | \
            (uint32_t)hashLookup[(d >> 24) & 0xff][0] << 16 | (uint32_t)hashLookup[(d >> 24) & 0xff][1] << 24; \
    b6 = (uint32_t)hashLookup[(d >> 0) & 0xff][0] | (uint32_t)hashLookup[(d >> 0) & 0xff][1] << 8 | \
            (uint32_t)hashLookup[(d >> 8) & 0xff][0] << 16 | (uint32_t)hashLookup[(d >> 8) & 0xff][1] << 24; \
}



/**
 * The loadPassword32 and storePassword32 methods are the preferred method for loading plains.
 * 
 * These work by loading the b0,b1,b2, etc directly from the memory space
 * as plaintext passwords.  At the end of each kernel execution, the current
 * passwords are stored back to the array.  This prevents the need to transfer
 * more plain start points to each thread when the kernel starts again.
 * 
 * @param pa Password initial array
 * @param dt Device number threads
 * @param pl Password length
 */
#define loadNTLMPasswords32(pa, dt, pl) { \
a = thread_index; \
b = pa[a]; \
b0 = (b & 0xff) | ((b & 0xff00) << 8); \
if (pl > 1) {b1 = ((b & 0xff0000) >> 16) | ((b & 0xff000000) >> 8);} \
if (pl > 3) {a += dt; b = pa[a]; b2 = (b & 0xff) | ((b & 0xff00) << 8);} \
if (pl > 5) {b3 = ((b & 0xff0000) >> 16) | ((b & 0xff000000) >> 8);} \
if (pl > 7) {a += dt; b = pa[a]; b4 = (b & 0xff) | ((b & 0xff00) << 8);} \
if (pl > 9) {b5 = ((b & 0xff0000) >> 16) | ((b & 0xff000000) >> 8);} \
if (pl > 11) {a += dt; b = pa[a]; b6 = (b & 0xff) | ((b & 0xff00) << 8);} \
if (pl > 13) {b7 = ((b & 0xff0000) >> 16) | ((b & 0xff000000) >> 8);} \
if (pl > 15) {a += dt; b = pa[a]; b8 = (b & 0xff) | ((b & 0xff00) << 8);} \
if (pl > 17) {b9 = ((b & 0xff0000) >> 16) | ((b & 0xff000000) >> 8);} \
if (pl > 19) {a += dt; b = pa[a]; b10 = (b & 0xff) | ((b & 0xff00) << 8);} \
if (pl > 21) {b11 = ((b & 0xff0000) >> 16) | ((b & 0xff000000) >> 8);} \
if (pl > 23) {a += dt; b = pa[a]; b12 = (b & 0xff) | ((b & 0xff00) << 8);} \
if (pl > 25) {b13 = ((b & 0xff0000) >> 16) | ((b & 0xff000000) >> 8);} \
}

#define storeNTLMPasswords32(pa, dt, pl) { \
b = (b0 & 0xff) | ((b0 & 0xff0000) >> 8); \
if (pl > 1) {b |= (b1 & 0xff) << 16 | ((b1 & 0xff0000) << 8);} \
pa[thread_index + 0] = b; \
if (pl > 3) {b = (b2 & 0xff) | ((b2 & 0xff0000) >> 8);} \
if (pl > 5) {b |= (b3 & 0xff) << 16 | ((b3 & 0xff0000) << 8);} \
if (pl > 3) {pa[thread_index + (dt * 1)] = b;} \
if (pl > 7) {b = (b4 & 0xff) | ((b4 & 0xff0000) >> 8);} \
if (pl > 9) {b |= (b5 & 0xff) << 16 | ((b5 & 0xff0000) << 8);} \
if (pl > 7) {pa[thread_index + (dt * 2)] = b;} \
if (pl > 11) {b = (b6 & 0xff) | ((b6 & 0xff0000) >> 8);} \
if (pl > 13) {b |= (b7 & 0xff) << 16 | ((b7 & 0xff0000) << 8);} \
if (pl > 11) {pa[thread_index + (dt * 3)] = b;} \
if (pl > 15) {b = (b8 & 0xff) | ((b8 & 0xff0000) >> 8);} \
if (pl > 17) {b |= (b9 & 0xff) << 16 | ((b9 & 0xff0000) << 8);} \
if (pl > 15) {pa[thread_index + (dt * 4)] = b;} \
if (pl > 19) {b = (b10 & 0xff) | ((b10 & 0xff0000) >> 8);} \
if (pl > 21) {b |= (b11 & 0xff) << 16 | ((b11 & 0xff0000) << 8);} \
if (pl > 19) {pa[thread_index + (dt * 5)] = b;} \
if (pl > 23) {b = (b12 & 0xff) | ((b12 & 0xff0000) >> 8);} \
if (pl > 25) {b |= (b13 & 0xff) << 16 | ((b13 & 0xff0000) << 8);} \
if (pl > 23) {pa[thread_index + (dt * 6)] = b;} \
}


/**
 * Searches for a 128 bit little endian NTLM hash in the global memory.
 *
 * This function takes the calculated hash values (a, b, c, d), the password
 * in b0, b1, etc (as NTLM style!), and the various global memory pointers
 * and searches for the hash.  If it is found, it reports it in the appropriate
 * method.
 *
 * @param a,b,c,d The calculated hash values.
 * @param b0-b7 The registers containing the input block in NTLM format
 * @param sharedBitmapA The address of the 8kb bitmap ideally in shared memory
 * @param deviceGlobalBitmap{A,B,C,D} The addresses (or null) of the device global bitmaps.
 * @param deviceGlobalFoundPasswords The address of the found-password array
 * @param deviceGlobalFoundPasswordFlags The address of the found-password flag array
 * @param deviceGlobalHashlistAddress The address of the 128-bit hash global hashlist
 * @param numberOfHashes The number of hashes being searched for currently
 * @param passwordLength The current password length
 */
__device__ inline void checkHash128LENTLM(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d,
        uint32_t &b0, uint32_t &b1, uint32_t &b2, uint32_t &b3, 
        uint32_t &b4, uint32_t &b5, uint32_t &b6, uint32_t &b7, 
        uint32_t &b8, uint32_t &b9, uint32_t &b10, uint32_t &b11, 
        uint32_t &b12, uint32_t &b13, 
        uint8_t *sharedBitmapA,
        uint8_t *deviceGlobalBitmapA, uint8_t *deviceGlobalBitmapB,
        uint8_t *deviceGlobalBitmapC, uint8_t *deviceGlobalBitmapD,
        uint8_t *deviceGlobalFoundPasswords, uint8_t *deviceGlobalFoundPasswordFlags,
        uint8_t *deviceGlobalHashlistAddress, uint64_t numberOfHashes,
        uint8_t passwordLength) {
    if ((sharedBitmapA[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) {
        if (!(deviceGlobalBitmapA) || ((deviceGlobalBitmapA[(a >> 3) & 0x07FFFFFF] >> (a & 0x7)) & 0x1)) {
            if (!deviceGlobalBitmapB || ((deviceGlobalBitmapB[(b >> 3) & 0x07FFFFFF] >> (b & 0x7)) & 0x1)) {
                if (!deviceGlobalBitmapC || ((deviceGlobalBitmapC[(c >> 3) & 0x07FFFFFF] >> (c & 0x7)) & 0x1)) {
                    if (!deviceGlobalBitmapD || ((deviceGlobalBitmapD[(d >> 3) & 0x07FFFFFF] >> (d & 0x7)) & 0x1)) {
                        uint32_t search_high, search_low, search_index, current_hash_value;
                        uint32_t *DEVICE_Hashes_32 = (uint32_t *) deviceGlobalHashlistAddress;
                        search_high = numberOfHashes;
                        search_low = 0;
                        while (search_low < search_high) {
                            // Midpoint between search_high and search_low
                            search_index = search_low + (search_high - search_low) / 2;
                            current_hash_value = DEVICE_Hashes_32[4 * search_index];
                            // Adjust search_high & search_low to work through space
                            if (current_hash_value < a) {
                                search_low = search_index + 1;
                            } else {
                                search_high = search_index;
                            }
                            if ((a == current_hash_value) && (search_low < numberOfHashes)) {
                                // Break out of the search loop - search_index is on a value
                                break;
                            }
                        }
                        // Broke out of the while loop

                        // If the loaded value does not match, there are no matches - just return.
                        if (a != current_hash_value) {
                            return;
                        }
                        // We've broken out of the loop, search_index should be on a matching value
                        // Loop while the search index is the same - linear search through this to find all possible
                        // matching passwords.
                        // We first need to move backwards to the beginning, as we may be in the middle of a set of matching hashes.
                        // If we are index 0, do NOT subtract, as we will wrap and this goes poorly.

                        while (search_index && (a == DEVICE_Hashes_32[(search_index - 1) * 4])) {
                            search_index--;
                        }
                        while ((a == DEVICE_Hashes_32[search_index * 4])) {
                            if (b == DEVICE_Hashes_32[search_index * 4 + 1]) {
                                if (c == DEVICE_Hashes_32[search_index * 4 + 2]) {
                                    if (d == DEVICE_Hashes_32[search_index * 4 + 3]) {
                                        // Copy the password to the correct location.
                                        switch (passwordLength) {
                                            case 27:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 26] = (b13 >> 0) & 0xff;
                                            case 26:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 25] = (b12 >> 16) & 0xff;
                                            case 25:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 24] = (b12 >> 0) & 0xff;
                                            case 24:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 23] = (b11 >> 16) & 0xff;
                                            case 23:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 22] = (b11 >> 0) & 0xff;
                                            case 22:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 21] = (b10 >> 16) & 0xff;
                                            case 21:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 20] = (b10 >> 0) & 0xff;
                                            case 20:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 19] = (b9 >> 16) & 0xff;
                                            case 19:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 18] = (b9 >> 0) & 0xff;
                                            case 18:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 17] = (b8 >> 16) & 0xff;
                                            case 17:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 16] = (b8 >> 0) & 0xff;
                                            case 16:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 15] = (b7 >> 16) & 0xff;
                                            case 15:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 14] = (b7 >> 0) & 0xff;
                                            case 14:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 13] = (b6 >> 16) & 0xff;
                                            case 13:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 12] = (b6 >> 0) & 0xff;
                                            case 12:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 11] = (b5 >> 16) & 0xff;
                                            case 11:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 10] = (b5 >> 0) & 0xff;
                                            case 10:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 9] = (b4 >> 16) & 0xff;
                                            case 9:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 8] = (b4 >> 0) & 0xff;
                                            case 8:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 7] = (b3 >> 16) & 0xff;
                                            case 7:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 6] = (b3 >> 0) & 0xff;
                                            case 6:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 5] = (b2 >> 16) & 0xff;
                                            case 5:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 4] = (b2 >> 0) & 0xff;
                                            case 4:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 3] = (b1 >> 16) & 0xff;
                                            case 3:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 2] = (b1 >> 0) & 0xff;
                                            case 2:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 1] = (b0 >> 16) & 0xff;
                                            case 1:
                                                deviceGlobalFoundPasswords[search_index * passwordLength + 0] = (b0 >> 0) & 0xff;
                                        }
                                        deviceGlobalFoundPasswordFlags[search_index] = (unsigned char) MFN_PASSWORD_NTLM;
                                    }
                                }
                            }
                            search_index++;
                        }
                    }
                }
            }
        }
    }
}



/**
 * This macro will store passwords from b0-b14 in a shared memory, per-block
 * array of uint32s.  This is used for algorithms that mangle the initial blocks
 * as part of their function (such as the SHA family).  There is a corresponding
 * function to load the values back (clearing other blocks as needed).  The
 * pass lengths are designed to store the final padding bit as well.
 * 
 * This function is designed for "packed" passwords - so anything but NTLM.
 * 
 * Note that this REQUIRES per-block memory!  It cannot be used with global
 * memory as it stores each thread's data in the same offset as all other
 * blocks.
 */
#define StoreNormalPasswordInShared(sharedArray, pass_length) { \
    sharedArray[threadIdx.x] = b0; \
    if (pass_length >  3) {sharedArray[threadIdx.x +  1 * blockDim.x] = b1;} \
    if (pass_length >  7) {sharedArray[threadIdx.x +  2 * blockDim.x] = b2;} \
    if (pass_length > 11) {sharedArray[threadIdx.x +  3 * blockDim.x] = b3;} \
    if (pass_length > 15) {sharedArray[threadIdx.x +  4 * blockDim.x] = b4;} \
    if (pass_length > 19) {sharedArray[threadIdx.x +  5 * blockDim.x] = b5;} \
    if (pass_length > 23) {sharedArray[threadIdx.x +  6 * blockDim.x] = b6;} \
    if (pass_length > 27) {sharedArray[threadIdx.x +  7 * blockDim.x] = b7;} \
    if (pass_length > 31) {sharedArray[threadIdx.x +  8 * blockDim.x] = b8;} \
    if (pass_length > 35) {sharedArray[threadIdx.x +  9 * blockDim.x] = b9;} \
    if (pass_length > 39) {sharedArray[threadIdx.x + 10 * blockDim.x] = b10;} \
    if (pass_length > 43) {sharedArray[threadIdx.x + 11 * blockDim.x] = b11;} \
    if (pass_length > 47) {sharedArray[threadIdx.x + 12 * blockDim.x] = b12;} \
    if (pass_length > 51) {sharedArray[threadIdx.x + 13 * blockDim.x] = b13;} \
}

/**
 * This function is the exact inverse of the above function.  It loads things
 * back into the registers from the shared array.  Note that this does not clear
 * any registers - if something has hashed them up, you MUST clear them before
 * calling this function!
 */
#define LoadNormalPasswordFromShared(sharedArray, pass_length) { \
    b0 = sharedArray[threadIdx.x]; \
    if (pass_length >  3) {b1  = sharedArray[threadIdx.x +  1 * blockDim.x];} \
    if (pass_length >  7) {b2  = sharedArray[threadIdx.x +  2 * blockDim.x];} \
    if (pass_length > 11) {b3  = sharedArray[threadIdx.x +  3 * blockDim.x];} \
    if (pass_length > 15) {b4  = sharedArray[threadIdx.x +  4 * blockDim.x];} \
    if (pass_length > 19) {b5  = sharedArray[threadIdx.x +  5 * blockDim.x];} \
    if (pass_length > 23) {b6  = sharedArray[threadIdx.x +  6 * blockDim.x];} \
    if (pass_length > 27) {b7  = sharedArray[threadIdx.x +  7 * blockDim.x];} \
    if (pass_length > 31) {b8  = sharedArray[threadIdx.x +  8 * blockDim.x];} \
    if (pass_length > 35) {b9  = sharedArray[threadIdx.x +  9 * blockDim.x];} \
    if (pass_length > 39) {b10 = sharedArray[threadIdx.x + 10 * blockDim.x];} \
    if (pass_length > 43) {b11 = sharedArray[threadIdx.x + 11 * blockDim.x];} \
    if (pass_length > 47) {b12 = sharedArray[threadIdx.x + 12 * blockDim.x];} \
    if (pass_length > 51) {b13 = sharedArray[threadIdx.x + 13 * blockDim.x];} \
}

/**
 * This function complements the above listed functions and expands a stored
 * password into the NTLM format.
 */
#define ExpandNTLMPasswordsFromShared(sharedArray, pass_length) { \
    b = sharedArray[threadIdx.x]; \
    b0 = (b & 0xff) | ((b & 0xff00) << 8); \
    if (pass_length > 1) { \
        b1 = ((b & 0xff0000) >> 16) | ((b & 0xff000000) >> 8); \
    } \
    if (pass_length > 3) { \
        b = sharedArray[threadIdx.x + 1 * blockDim.x]; \
        b2 = (b & 0xff) | ((b & 0xff00) << 8); \
    } \
    if (pass_length > 5) { \
        b3 = ((b & 0xff0000) >> 16) | ((b & 0xff000000) >> 8); \
    } \
    if (pass_length > 7) { \
        b = sharedArray[threadIdx.x + 2 * blockDim.x]; \
        b4 = (b & 0xff) | ((b & 0xff00) << 8); \
    } \
    if (pass_length > 9) { \
        b5 = ((b & 0xff0000) >> 16) | ((b & 0xff000000) >> 8); \
    } \
    if (pass_length > 11) { \
        b = sharedArray[threadIdx.x + 3 * blockDim.x]; \
        b6 = (b & 0xff) | ((b & 0xff00) << 8); \
    } \
    if (pass_length > 13) { \
        b7 = ((b & 0xff0000) >> 16) | ((b & 0xff000000) >> 8); \
    } \
    if (pass_length > 15) { \
        b = sharedArray[threadIdx.x + 4 * blockDim.x]; \
        b8 = (b & 0xff) | ((b & 0xff00) << 8); \
    } \
    if (pass_length > 17) { \
        b9 = ((b & 0xff0000) >> 16) | ((b & 0xff000000) >> 8); \
    } \
    if (pass_length > 19) { \
        b = sharedArray[threadIdx.x + 5 * blockDim.x]; \
        b10 = (b & 0xff) | ((b & 0xff00) << 8); \
    } \
    if (pass_length > 21) { \
        b11 = ((b & 0xff0000) >> 16) | ((b & 0xff000000) >> 8); \
    } \
    if (pass_length > 23) { \
        b = sharedArray[threadIdx.x + 6 * blockDim.x]; \
        b12 = (b & 0xff) | ((b & 0xff00) << 8); \
    } \
    if (pass_length > 25) { \
        b13 = ((b & 0xff0000) >> 16) | ((b & 0xff000000) >> 8); \
    } \
}

#endif
