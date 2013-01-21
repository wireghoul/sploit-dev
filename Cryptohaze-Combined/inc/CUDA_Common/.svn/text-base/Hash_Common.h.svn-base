/*
Cryptohaze GPU Rainbow Tables
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

// This file contains various useful things for hashes...

#ifndef __CUDA_HASH_COMMON_H
#define	__CUDA_HASH_COMMON_H

// This is here so Netbeans doesn't error-spam my IDE
#if !defined(__CUDACC__)
    // define the keywords, so that the IDE does not complain about them
    #define __global__
    #define __device__
    #define __shared__
    #define __constant__
    #define blockIdx.x 1
    #define blockDim.x 1
    #define threadIdx.x 1

    // Constants used in here.
    #define Device_Table_Index
    #define Device_Charset_Constant
#endif

// If the UseB0-B13 stuff is not defined, use all of them.
#ifndef UseB0
    #define UseB0 1
    #define UseB1 1
    #define UseB2 1
    #define UseB3 1
    #define UseB4 1
    #define UseB5 1
    #define UseB6 1
    #define UseB7 1
    #define UseB8 1
    #define UseB9 1
    #define UseB10 1
    #define UseB11 1
    #define UseB12 1
    #define UseB13 1
#endif

__device__ inline void clearB0toB15(uint32_t &b0, uint32_t &b1, uint32_t &b2, uint32_t &b3, uint32_t &b4, uint32_t &b5, uint32_t &b6, uint32_t &b7,
			   uint32_t &b8, uint32_t &b9, uint32_t &b10, uint32_t &b11, uint32_t &b12, uint32_t &b13, uint32_t &b14, uint32_t &b15) {
  b0 = 0x00000000;
  b1 = 0x00000000;
  b2 = 0x00000000;
  b3 = 0x00000000;
  b4 = 0x00000000;
  b5 = 0x00000000;
  b6 = 0x00000000;
  b7 = 0x00000000;
  b8 = 0x00000000;
  b9 = 0x00000000;
  b10 = 0x00000000;
  b11 = 0x00000000;
  b12 = 0x00000000;
  b13 = 0x00000000;
  b14 = 0x00000000;
  b15 = 0x00000000;
}


// padMDHash sets up the length and padding for a MD{4,5} hash.
__device__ inline void padMDHash(int length,
                uint32_t &b0, uint32_t &b1, uint32_t &b2, uint32_t &b3, uint32_t &b4, uint32_t &b5, uint32_t &b6, uint32_t &b7,
		uint32_t &b8, uint32_t &b9, uint32_t &b10, uint32_t &b11, uint32_t &b12, uint32_t &b13, uint32_t &b14, uint32_t &b15) {

  // Set length properly (length in bits)
  b14 = length * 8;

  if (length == 0) {
    b0 |= 0x00000080;
    return;
  }
  if (length == 1) {
    b0 |= 0x00008000;
    return;
  }
  if (length == 2) {
    b0 |= 0x00800000;
    return;
  }
  if (length == 3) {
    b0 |= 0x80000000;
    return;
  }
  if (length == 4) {
    b1 |= 0x00000080;
    return;
  }
  if (length == 5) {
    b1 |= 0x00008000;
    return;
  }
  if (length == 6) {
    b1 |= 0x00800000;
    return;
  }
  if (length == 7) {
    b1 |= 0x80000000;
    return;
  }
  if (length == 8) {
    b2 |= 0x00000080;
    return;
  }
  if (length == 9) {
    b2 |= 0x00008000;
    return;
  }
  if (length == 10) {
    b2 |= 0x00800000;
    return;
  }
  if (length == 11) {
   b2 |= 0x80000000;
    return;
  }
  if (length == 12) {
    b3 |= 0x00000080;
    return;
  }
  if (length == 13) {
    b3 |= 0x00008000;
    return;
  }
  if (length == 14) {
    b3 |= 0x00800000;
    return;
  }
  if (length == 15) {
    b3 |= 0x80000000;
    return;
  }
  if (length == 16) {
    b4 |= 0x00000080;
    return;
  }
  if (length == 17) {
    b4 |= 0x00008000;
    return;
  }
  if (length == 18) {
    b4 |= 0x00800000;
    return;
  }
  if (length == 19) {
    b4 |= 0x80000000;
    return;
  }
  if (length == 20) {
    b5 |= 0x00000080;
    return;
  }
  if (length == 21) {
    b5 |= 0x00008000;
    return;
  }
  if (length == 22) {
    b5 |= 0x00800000;
    return;
  }
  if (length == 23) {
    b5 |= 0x80000000;
    return;
  }

}


// Copy charset into shared memory.
__device__ inline void copySingleCharsetToShared(char *sharedCharset, unsigned char *constantCharset) {

    // 32-bit accesses are used to help coalesce memory accesses.
    uint32_t a;
    uint32_t *CharsetAccess32, *CharsetConstant32;
    CharsetAccess32 = (uint32_t *)sharedCharset;
    // Device_Charset_Constant is in constant memory
    CharsetConstant32 = (uint32_t *)constantCharset;

    for (a = 0; a < (512 / sizeof(uint32_t)); a++) {
        CharsetAccess32[a] = CharsetConstant32[a];
    }
    syncthreads();
}


// Sets the character at the given position.  Requires things to be zeroed first!
__device__ inline void SetCharacterAtPosition(unsigned char character, unsigned char position,
        uint32_t &b0, uint32_t &b1, uint32_t &b2, uint32_t &b3, uint32_t &b4, uint32_t &b5, uint32_t &b6, uint32_t &b7,
	uint32_t &b8, uint32_t &b9, uint32_t &b10, uint32_t &b11, uint32_t &b12, uint32_t &b13, uint32_t &b14, uint32_t &b15) {

    int offset = position / 4;

    if (UseB0 & offset == 0) {
        b0 |= character << (8 * (position % 4));
    } else if (UseB1 & offset == 1) {
        b1 |= character << (8 * (position % 4));
    } else if (UseB2 & offset == 2) {
        b2 |= character << (8 * (position % 4));
    } else if (UseB3 & offset == 3) {
        b3 |= character << (8 * (position % 4));
    } else if (UseB4 & offset == 4) {
        b4 |= character << (8 * (position % 4));
    } else if (UseB5 & offset == 5) {
        b5 |= character << (8 * (position % 4));
    } else if (UseB6 & offset == 6) {
        b6 |= character << (8 * (position % 4));
    } else if (UseB7 & offset == 7) {
        b7 |= character << (8 * (position % 4));
    } else if (UseB8 & offset == 8) {
        b8 |= character << (8 * (position % 4));
    } else if (UseB9 & offset == 9) {
        b9 |= character << (8 * (position % 4));
    } else if (UseB10 & offset == 10) {
        b10 |= character << (8 * (position % 4));
    } else if (UseB11 & offset == 11) {
        b11 |= character << (8 * (position % 4));
    } else if (UseB12 & offset == 12) {
        b12 |= character << (8 * (position % 4));
    } else if (UseB13 & offset == 13) {
        b13 |= character << (8 * (position % 4));
    }
}



#endif
