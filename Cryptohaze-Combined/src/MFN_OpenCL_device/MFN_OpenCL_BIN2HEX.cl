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
 * This file contains utilities for converting binary data to ascii hex, as is
 * done in many PHP scripts for web authentication.
 * 
 * Big endian and little endian hashes are all very different!  Be sure you are
 * using the correct function!
 */

/**
 * Character values to load into the shared memory lookup tables.
 */
__constant unsigned char hexLookupValues[16] = 
    {'0', '1', '2', '3', '4', '5', '6', '7', 
    '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};


/**
 * Convert data from a little endian hash (MD5) into a little endian hash (MD5).
 * This is used for double MD5 and similar.
 * 
 * This function takes the lower 16 bits in hashValue and converts them to ASCII
 * in blockValue.  It then shifts hashValue right by 16 bits in preparation for
 * future calls to this (for the next block).  See DoubleMD5 for an example.
 */
#if grt_vector_1
#define AddHashCharacterAsString_LE_LE(hashLookup, hashValue, blockValue, temp) { \
  temp = (hashValue >> 8) & 0xff; \
  blockValue |= hashLookup[temp & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue |= hashLookup[temp & 0xff][0]; \
  blockValue = blockValue << 8; \
  temp = (hashValue) & 0xff; \
  blockValue |= hashLookup[temp & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue |= hashLookup[temp & 0xff][0]; \
  hashValue = hashValue >> 16; \
}
#elif grt_vector_2
#define AddHashCharacterAsString_LE_LE(hashLookup, hashValue, blockValue, temp) { \
  temp = (hashValue >> 8) & 0xff; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][0]; \
  blockValue = blockValue << 8; \
  temp = (hashValue) & 0xff; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][0]; \
  hashValue = hashValue >> 16; \
}
#elif grt_vector_4
#define AddHashCharacterAsString_LE_LE(hashLookup, hashValue, blockValue, temp) { \
  temp = (hashValue >> 8) & 0xff; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][1]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][1]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][0]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][0]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][0]; \
  blockValue = blockValue << 8; \
  temp = (hashValue) & 0xff; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][1]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][1]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][0]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][0]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][0]; \
  hashValue = hashValue >> 16; \
}
#elif grt_vector_8
#define AddHashCharacterAsString_LE_LE(hashLookup, hashValue, blockValue, temp) { \
  temp = (hashValue >> 8) & 0xff; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][1]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][1]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][1]; \
  blockValue.s4 |= hashLookup[temp.s4 & 0xff][1]; \
  blockValue.s5 |= hashLookup[temp.s5 & 0xff][1]; \
  blockValue.s6 |= hashLookup[temp.s6 & 0xff][1]; \
  blockValue.s7 |= hashLookup[temp.s7 & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][0]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][0]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][0]; \
  blockValue.s4 |= hashLookup[temp.s4 & 0xff][0]; \
  blockValue.s5 |= hashLookup[temp.s5 & 0xff][0]; \
  blockValue.s6 |= hashLookup[temp.s6 & 0xff][0]; \
  blockValue.s7 |= hashLookup[temp.s7 & 0xff][0]; \
  blockValue = blockValue << 8; \
  temp = (hashValue) & 0xff; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][1]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][1]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][1]; \
  blockValue.s4 |= hashLookup[temp.s4 & 0xff][1]; \
  blockValue.s5 |= hashLookup[temp.s5 & 0xff][1]; \
  blockValue.s6 |= hashLookup[temp.s6 & 0xff][1]; \
  blockValue.s7 |= hashLookup[temp.s7 & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][0]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][0]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][0]; \
  blockValue.s4 |= hashLookup[temp.s4 & 0xff][0]; \
  blockValue.s5 |= hashLookup[temp.s5 & 0xff][0]; \
  blockValue.s6 |= hashLookup[temp.s6 & 0xff][0]; \
  blockValue.s7 |= hashLookup[temp.s7 & 0xff][0]; \
  hashValue = hashValue >> 16; \
}
#elif grt_vector_16
#define AddHashCharacterAsString_LE_LE(hashLookup, hashValue, blockValue, temp) { \
  temp = (hashValue >> 8) & 0xff; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][1]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][1]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][1]; \
  blockValue.s4 |= hashLookup[temp.s4 & 0xff][1]; \
  blockValue.s5 |= hashLookup[temp.s5 & 0xff][1]; \
  blockValue.s6 |= hashLookup[temp.s6 & 0xff][1]; \
  blockValue.s7 |= hashLookup[temp.s7 & 0xff][1]; \
  blockValue.s8 |= hashLookup[temp.s8 & 0xff][1]; \
  blockValue.s9 |= hashLookup[temp.s9 & 0xff][1]; \
  blockValue.sA |= hashLookup[temp.sA & 0xff][1]; \
  blockValue.sB |= hashLookup[temp.sB & 0xff][1]; \
  blockValue.sC |= hashLookup[temp.sC & 0xff][1]; \
  blockValue.sD |= hashLookup[temp.sD & 0xff][1]; \
  blockValue.sE |= hashLookup[temp.sE & 0xff][1]; \
  blockValue.sF |= hashLookup[temp.sF & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][0]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][0]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][0]; \
  blockValue.s4 |= hashLookup[temp.s4 & 0xff][0]; \
  blockValue.s5 |= hashLookup[temp.s5 & 0xff][0]; \
  blockValue.s6 |= hashLookup[temp.s6 & 0xff][0]; \
  blockValue.s7 |= hashLookup[temp.s7 & 0xff][0]; \
  blockValue.s8 |= hashLookup[temp.s8 & 0xff][0]; \
  blockValue.s9 |= hashLookup[temp.s9 & 0xff][0]; \
  blockValue.sA |= hashLookup[temp.sA & 0xff][0]; \
  blockValue.sB |= hashLookup[temp.sB & 0xff][0]; \
  blockValue.sC |= hashLookup[temp.sC & 0xff][0]; \
  blockValue.sD |= hashLookup[temp.sD & 0xff][0]; \
  blockValue.sE |= hashLookup[temp.sE & 0xff][0]; \
  blockValue.sF |= hashLookup[temp.sF & 0xff][0]; \
  blockValue = blockValue << 8; \
  temp = (hashValue) & 0xff; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][1]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][1]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][1]; \
  blockValue.s4 |= hashLookup[temp.s4 & 0xff][1]; \
  blockValue.s5 |= hashLookup[temp.s5 & 0xff][1]; \
  blockValue.s6 |= hashLookup[temp.s6 & 0xff][1]; \
  blockValue.s7 |= hashLookup[temp.s7 & 0xff][1]; \
  blockValue.s8 |= hashLookup[temp.s8 & 0xff][1]; \
  blockValue.s9 |= hashLookup[temp.s9 & 0xff][1]; \
  blockValue.sA |= hashLookup[temp.sA & 0xff][1]; \
  blockValue.sB |= hashLookup[temp.sB & 0xff][1]; \
  blockValue.sC |= hashLookup[temp.sC & 0xff][1]; \
  blockValue.sD |= hashLookup[temp.sD & 0xff][1]; \
  blockValue.sE |= hashLookup[temp.sE & 0xff][1]; \
  blockValue.sF |= hashLookup[temp.sF & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][0]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][0]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][0]; \
  blockValue.s4 |= hashLookup[temp.s4 & 0xff][0]; \
  blockValue.s5 |= hashLookup[temp.s5 & 0xff][0]; \
  blockValue.s6 |= hashLookup[temp.s6 & 0xff][0]; \
  blockValue.s7 |= hashLookup[temp.s7 & 0xff][0]; \
  blockValue.s8 |= hashLookup[temp.s8 & 0xff][0]; \
  blockValue.s9 |= hashLookup[temp.s9 & 0xff][0]; \
  blockValue.sA |= hashLookup[temp.sA & 0xff][0]; \
  blockValue.sB |= hashLookup[temp.sB & 0xff][0]; \
  blockValue.sC |= hashLookup[temp.sC & 0xff][0]; \
  blockValue.sD |= hashLookup[temp.sD & 0xff][0]; \
  blockValue.sE |= hashLookup[temp.sE & 0xff][0]; \
  blockValue.sF |= hashLookup[temp.sF & 0xff][0]; \
  hashValue = hashValue >> 16; \
}
#endif



/**
 * Convert data from a little endian hash (MD5) into a little endian hash (MD5).
 * This is used for double MD5 and similar, with the difference that it is a 
 * "vector expand" function - it takes the value in one vector element and loads
 * all the vector elements of the block with it.  Used for salt loading in IPB
 * among other places.
 */
#if grt_vector_1
#define AddHashCharacterAsString_LE_LE_VE(hashLookup, hashValue, blockValue) { \
  { \
  uint temp = (hashValue >> 8) & 0xff; \
  blockValue |= hashLookup[temp & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue |= hashLookup[temp & 0xff][0]; \
  blockValue = blockValue << 8; \
  temp = (hashValue) & 0xff; \
  blockValue |= hashLookup[temp & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue |= hashLookup[temp & 0xff][0]; \
  hashValue = hashValue >> 16; \
  } \
}
#elif grt_vector_2
#define AddHashCharacterAsString_LE_LE_VE(hashLookup, hashValue, blockValue) { \
  { \
  uint temp = (hashValue >> 8) & 0xff; \
  blockValue.s0 |= hashLookup[temp & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp & 0xff][0]; \
  blockValue = blockValue << 8; \
  temp = (hashValue) & 0xff; \
  blockValue.s0 |= hashLookup[temp & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp & 0xff][0]; \
  hashValue = hashValue >> 16; \
  } \
}
#elif grt_vector_4
#define AddHashCharacterAsString_LE_LE_VE(hashLookup, hashValue, blockValue) { \
  { \
  uint temp = (hashValue >> 8) & 0xff; \
  blockValue.s0 |= hashLookup[temp & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp & 0xff][1]; \
  blockValue.s2 |= hashLookup[temp & 0xff][1]; \
  blockValue.s3 |= hashLookup[temp & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp & 0xff][0]; \
  blockValue.s2 |= hashLookup[temp & 0xff][0]; \
  blockValue.s3 |= hashLookup[temp & 0xff][0]; \
  blockValue = blockValue << 8; \
  temp = (hashValue) & 0xff; \
  blockValue.s0 |= hashLookup[temp & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp & 0xff][1]; \
  blockValue.s2 |= hashLookup[temp & 0xff][1]; \
  blockValue.s3 |= hashLookup[temp & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp & 0xff][0]; \
  blockValue.s2 |= hashLookup[temp & 0xff][0]; \
  blockValue.s3 |= hashLookup[temp & 0xff][0]; \
  hashValue = hashValue >> 16; \
  } \
}
#elif grt_vector_8
#define AddHashCharacterAsString_LE_LE_VE(hashLookup, hashValue, blockValue) { \
  { \
  uint temp = (hashValue >> 8) & 0xff; \
  blockValue.s0 |= hashLookup[temp & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp & 0xff][1]; \
  blockValue.s2 |= hashLookup[temp & 0xff][1]; \
  blockValue.s3 |= hashLookup[temp & 0xff][1]; \
  blockValue.s4 |= hashLookup[temp & 0xff][1]; \
  blockValue.s5 |= hashLookup[temp & 0xff][1]; \
  blockValue.s6 |= hashLookup[temp & 0xff][1]; \
  blockValue.s7 |= hashLookup[temp & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp & 0xff][0]; \
  blockValue.s2 |= hashLookup[temp & 0xff][0]; \
  blockValue.s3 |= hashLookup[temp & 0xff][0]; \
  blockValue.s4 |= hashLookup[temp & 0xff][0]; \
  blockValue.s5 |= hashLookup[temp & 0xff][0]; \
  blockValue.s6 |= hashLookup[temp & 0xff][0]; \
  blockValue.s7 |= hashLookup[temp & 0xff][0]; \
  blockValue = blockValue << 8; \
  temp = (hashValue) & 0xff; \
  blockValue.s0 |= hashLookup[temp & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp & 0xff][1]; \
  blockValue.s2 |= hashLookup[temp & 0xff][1]; \
  blockValue.s3 |= hashLookup[temp & 0xff][1]; \
  blockValue.s4 |= hashLookup[temp & 0xff][1]; \
  blockValue.s5 |= hashLookup[temp & 0xff][1]; \
  blockValue.s6 |= hashLookup[temp & 0xff][1]; \
  blockValue.s7 |= hashLookup[temp & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp & 0xff][0]; \
  blockValue.s2 |= hashLookup[temp & 0xff][0]; \
  blockValue.s3 |= hashLookup[temp & 0xff][0]; \
  blockValue.s4 |= hashLookup[temp & 0xff][0]; \
  blockValue.s5 |= hashLookup[temp & 0xff][0]; \
  blockValue.s6 |= hashLookup[temp & 0xff][0]; \
  blockValue.s7 |= hashLookup[temp & 0xff][0]; \
  hashValue = hashValue >> 16; \
  } \
}
#elif grt_vector_16
#define AddHashCharacterAsString_LE_LE_VE(hashLookup, hashValue, blockValue) { \
  { \
  uint temp = (hashValue >> 8) & 0xff; \
  blockValue.s0 |= hashLookup[temp & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp & 0xff][1]; \
  blockValue.s2 |= hashLookup[temp & 0xff][1]; \
  blockValue.s3 |= hashLookup[temp & 0xff][1]; \
  blockValue.s4 |= hashLookup[temp & 0xff][1]; \
  blockValue.s5 |= hashLookup[temp & 0xff][1]; \
  blockValue.s6 |= hashLookup[temp & 0xff][1]; \
  blockValue.s7 |= hashLookup[temp & 0xff][1]; \
  blockValue.s8 |= hashLookup[temp & 0xff][1]; \
  blockValue.s9 |= hashLookup[temp & 0xff][1]; \
  blockValue.sA |= hashLookup[temp & 0xff][1]; \
  blockValue.sB |= hashLookup[temp & 0xff][1]; \
  blockValue.sC |= hashLookup[temp & 0xff][1]; \
  blockValue.sD |= hashLookup[temp & 0xff][1]; \
  blockValue.sE |= hashLookup[temp & 0xff][1]; \
  blockValue.sF |= hashLookup[temp & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp & 0xff][0]; \
  blockValue.s2 |= hashLookup[temp & 0xff][0]; \
  blockValue.s3 |= hashLookup[temp & 0xff][0]; \
  blockValue.s4 |= hashLookup[temp & 0xff][0]; \
  blockValue.s5 |= hashLookup[temp & 0xff][0]; \
  blockValue.s6 |= hashLookup[temp & 0xff][0]; \
  blockValue.s7 |= hashLookup[temp & 0xff][0]; \
  blockValue.s8 |= hashLookup[temp & 0xff][0]; \
  blockValue.s9 |= hashLookup[temp & 0xff][0]; \
  blockValue.sA |= hashLookup[temp & 0xff][0]; \
  blockValue.sB |= hashLookup[temp & 0xff][0]; \
  blockValue.sC |= hashLookup[temp & 0xff][0]; \
  blockValue.sD |= hashLookup[temp & 0xff][0]; \
  blockValue.sE |= hashLookup[temp & 0xff][0]; \
  blockValue.sF |= hashLookup[temp & 0xff][0]; \
  blockValue = blockValue << 8; \
  temp = (hashValue) & 0xff; \
  blockValue.s0 |= hashLookup[temp & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp & 0xff][1]; \
  blockValue.s2 |= hashLookup[temp & 0xff][1]; \
  blockValue.s3 |= hashLookup[temp & 0xff][1]; \
  blockValue.s4 |= hashLookup[temp & 0xff][1]; \
  blockValue.s5 |= hashLookup[temp & 0xff][1]; \
  blockValue.s6 |= hashLookup[temp & 0xff][1]; \
  blockValue.s7 |= hashLookup[temp & 0xff][1]; \
  blockValue.s8 |= hashLookup[temp & 0xff][1]; \
  blockValue.s9 |= hashLookup[temp & 0xff][1]; \
  blockValue.sA |= hashLookup[temp & 0xff][1]; \
  blockValue.sB |= hashLookup[temp & 0xff][1]; \
  blockValue.sC |= hashLookup[temp & 0xff][1]; \
  blockValue.sD |= hashLookup[temp & 0xff][1]; \
  blockValue.sE |= hashLookup[temp & 0xff][1]; \
  blockValue.sF |= hashLookup[temp & 0xff][1]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp & 0xff][0]; \
  blockValue.s2 |= hashLookup[temp & 0xff][0]; \
  blockValue.s3 |= hashLookup[temp & 0xff][0]; \
  blockValue.s4 |= hashLookup[temp & 0xff][0]; \
  blockValue.s5 |= hashLookup[temp & 0xff][0]; \
  blockValue.s6 |= hashLookup[temp & 0xff][0]; \
  blockValue.s7 |= hashLookup[temp & 0xff][0]; \
  blockValue.s8 |= hashLookup[temp & 0xff][0]; \
  blockValue.s9 |= hashLookup[temp & 0xff][0]; \
  blockValue.sA |= hashLookup[temp & 0xff][0]; \
  blockValue.sB |= hashLookup[temp & 0xff][0]; \
  blockValue.sC |= hashLookup[temp & 0xff][0]; \
  blockValue.sD |= hashLookup[temp & 0xff][0]; \
  blockValue.sE |= hashLookup[temp & 0xff][0]; \
  blockValue.sF |= hashLookup[temp & 0xff][0]; \
  hashValue = hashValue >> 16; \
  } \
}
#endif


/**
 * This does the same as above, but without using any memory.
 */
#define AddHashCharacterAsString_LE_LE_NoMem(hashValue, blockVariable) { \
    blockVariable = ((hashValue & 0xf0) >> 4) | ((hashValue & 0x0f) << 8) | ((hashValue & 0xf000) << 4) | ((hashValue & 0x0f00) << 16); \
    blockVariable  += 0x30303030; \
    blockVariable  += ((blockVariable  & 0x000000ff) > 0x00000039) ? (vector_type)0x00000027 : (vector_type)0x00000000; \
    blockVariable  += ((blockVariable  & 0x0000ff00) > 0x00003900) ? (vector_type)0x00002700 : (vector_type)0x00000000; \
    blockVariable  += ((blockVariable  & 0x00ff0000) > 0x00390000) ? (vector_type)0x00270000 : (vector_type)0x00000000; \
    blockVariable  += ((blockVariable  & 0xff000000) > 0x39000000) ? (vector_type)0x27000000 : (vector_type)0x00000000; \
    hashValue = hashValue >> 16; \
}

/**
 * Load a big endian hash (SHA1, SHA256, etc) as a big endian hash.  This is
 * for double SHA hashes, and various web hashes that require loading a SHA
 * function output as ASCII.
 */
#if grt_vector_1
#define AddHashCharacterAsString_BE_BE(hashLookup, hashValue, blockValue, temp) { \
  temp = (hashValue >> 24) & 0xff; \
  blockValue |= hashLookup[temp & 0xff][0]; \
  blockValue = blockValue << 8; \
  blockValue |= hashLookup[temp & 0xff][1]; \
  blockValue = blockValue << 8; \
  temp = (hashValue >> 16) & 0xff; \
  blockValue |= hashLookup[temp & 0xff][0]; \
  blockValue = blockValue << 8; \
  blockValue |= hashLookup[temp & 0xff][1]; \
  hashValue = hashValue << 16; \
}
#elif grt_vector_2
#define AddHashCharacterAsString_BE_BE(hashLookup, hashValue, blockValue, temp) { \
  temp = (hashValue >> 24) & 0xff; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][0]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][1]; \
  blockValue = blockValue << 8; \
  temp = (hashValue >> 16) & 0xff; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][0]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][1]; \
  hashValue = hashValue << 16; \
}
#elif grt_vector_4
#define AddHashCharacterAsString_BE_BE(hashLookup, hashValue, blockValue, temp) { \
  temp = (hashValue >> 24) & 0xff; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][0]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][0]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][0]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][1]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][1]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][1]; \
  blockValue = blockValue << 8; \
  temp = (hashValue >> 16) & 0xff; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][0]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][0]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][0]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][1]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][1]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][1]; \
  hashValue = hashValue << 16; \
}
#elif grt_vector_8
#define AddHashCharacterAsString_BE_BE(hashLookup, hashValue, blockValue, temp) { \
  temp = (hashValue >> 24) & 0xff; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][0]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][0]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][0]; \
  blockValue.s4 |= hashLookup[temp.s4 & 0xff][0]; \
  blockValue.s5 |= hashLookup[temp.s5 & 0xff][0]; \
  blockValue.s6 |= hashLookup[temp.s6 & 0xff][0]; \
  blockValue.s7 |= hashLookup[temp.s7 & 0xff][0]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][1]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][1]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][1]; \
  blockValue.s4 |= hashLookup[temp.s4 & 0xff][1]; \
  blockValue.s5 |= hashLookup[temp.s5 & 0xff][1]; \
  blockValue.s6 |= hashLookup[temp.s6 & 0xff][1]; \
  blockValue.s7 |= hashLookup[temp.s7 & 0xff][1]; \
  blockValue = blockValue << 8; \
  temp = (hashValue >> 16) & 0xff; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][0]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][0]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][0]; \
  blockValue.s4 |= hashLookup[temp.s4 & 0xff][0]; \
  blockValue.s5 |= hashLookup[temp.s5 & 0xff][0]; \
  blockValue.s6 |= hashLookup[temp.s6 & 0xff][0]; \
  blockValue.s7 |= hashLookup[temp.s7 & 0xff][0]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][1]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][1]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][1]; \
  blockValue.s4 |= hashLookup[temp.s4 & 0xff][1]; \
  blockValue.s5 |= hashLookup[temp.s5 & 0xff][1]; \
  blockValue.s6 |= hashLookup[temp.s6 & 0xff][1]; \
  blockValue.s7 |= hashLookup[temp.s7 & 0xff][1]; \
  hashValue = hashValue << 16; \
}
#elif grt_vector_16
#define AddHashCharacterAsString_BE_BE(hashLookup, hashValue, blockValue, temp) { \
  temp = (hashValue >> 24) & 0xff; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][0]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][0]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][0]; \
  blockValue.s4 |= hashLookup[temp.s4 & 0xff][0]; \
  blockValue.s5 |= hashLookup[temp.s5 & 0xff][0]; \
  blockValue.s6 |= hashLookup[temp.s6 & 0xff][0]; \
  blockValue.s7 |= hashLookup[temp.s7 & 0xff][0]; \
  blockValue.s8 |= hashLookup[temp.s8 & 0xff][0]; \
  blockValue.s9 |= hashLookup[temp.s9 & 0xff][0]; \
  blockValue.sA |= hashLookup[temp.sA & 0xff][0]; \
  blockValue.sB |= hashLookup[temp.sB & 0xff][0]; \
  blockValue.sC |= hashLookup[temp.sC & 0xff][0]; \
  blockValue.sD |= hashLookup[temp.sD & 0xff][0]; \
  blockValue.sE |= hashLookup[temp.sE & 0xff][0]; \
  blockValue.sF |= hashLookup[temp.sF & 0xff][0]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][1]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][1]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][1]; \
  blockValue.s4 |= hashLookup[temp.s4 & 0xff][1]; \
  blockValue.s5 |= hashLookup[temp.s5 & 0xff][1]; \
  blockValue.s6 |= hashLookup[temp.s6 & 0xff][1]; \
  blockValue.s7 |= hashLookup[temp.s7 & 0xff][1]; \
  blockValue.s8 |= hashLookup[temp.s8 & 0xff][1]; \
  blockValue.s9 |= hashLookup[temp.s9 & 0xff][1]; \
  blockValue.sA |= hashLookup[temp.sA & 0xff][1]; \
  blockValue.sB |= hashLookup[temp.sB & 0xff][1]; \
  blockValue.sC |= hashLookup[temp.sC & 0xff][1]; \
  blockValue.sD |= hashLookup[temp.sD & 0xff][1]; \
  blockValue.sE |= hashLookup[temp.sE & 0xff][1]; \
  blockValue.sF |= hashLookup[temp.sF & 0xff][1]; \
  blockValue = blockValue << 8; \
  temp = (hashValue >> 16) & 0xff; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][0]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][0]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][0]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][0]; \
  blockValue.s4 |= hashLookup[temp.s4 & 0xff][0]; \
  blockValue.s5 |= hashLookup[temp.s5 & 0xff][0]; \
  blockValue.s6 |= hashLookup[temp.s6 & 0xff][0]; \
  blockValue.s7 |= hashLookup[temp.s7 & 0xff][0]; \
  blockValue.s8 |= hashLookup[temp.s8 & 0xff][0]; \
  blockValue.s9 |= hashLookup[temp.s9 & 0xff][0]; \
  blockValue.sA |= hashLookup[temp.sA & 0xff][0]; \
  blockValue.sB |= hashLookup[temp.sB & 0xff][0]; \
  blockValue.sC |= hashLookup[temp.sC & 0xff][0]; \
  blockValue.sD |= hashLookup[temp.sD & 0xff][0]; \
  blockValue.sE |= hashLookup[temp.sE & 0xff][0]; \
  blockValue.sF |= hashLookup[temp.sF & 0xff][0]; \
  blockValue = blockValue << 8; \
  blockValue.s0 |= hashLookup[temp.s0 & 0xff][1]; \
  blockValue.s1 |= hashLookup[temp.s1 & 0xff][1]; \
  blockValue.s2 |= hashLookup[temp.s2 & 0xff][1]; \
  blockValue.s3 |= hashLookup[temp.s3 & 0xff][1]; \
  blockValue.s4 |= hashLookup[temp.s4 & 0xff][1]; \
  blockValue.s5 |= hashLookup[temp.s5 & 0xff][1]; \
  blockValue.s6 |= hashLookup[temp.s6 & 0xff][1]; \
  blockValue.s7 |= hashLookup[temp.s7 & 0xff][1]; \
  blockValue.s8 |= hashLookup[temp.s8 & 0xff][1]; \
  blockValue.s9 |= hashLookup[temp.s9 & 0xff][1]; \
  blockValue.sA |= hashLookup[temp.sA & 0xff][1]; \
  blockValue.sB |= hashLookup[temp.sB & 0xff][1]; \
  blockValue.sC |= hashLookup[temp.sC & 0xff][1]; \
  blockValue.sD |= hashLookup[temp.sD & 0xff][1]; \
  blockValue.sE |= hashLookup[temp.sE & 0xff][1]; \
  blockValue.sF |= hashLookup[temp.sF & 0xff][1]; \
  hashValue = hashValue << 16; \
}
#endif


// Macro template to expand.
#if grt_vector_1
#elif grt_vector_2
#elif grt_vector_4
#elif grt_vector_8
#elif grt_vector_16
#endif
