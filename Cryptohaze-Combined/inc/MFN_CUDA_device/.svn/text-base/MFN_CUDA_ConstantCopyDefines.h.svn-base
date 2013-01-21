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
 * This file contains defines for the various values that are copied to the GPU
 * in constant memory.  This is needed because CUDA5 eliminated the ability to
 * use strings to describe them, and instead requires direct variable reference.
 * 
 * Instead of doing very ugly things, this allows the replacement of the
 * existing constant copy function calls with a slightly modified version.  The
 * host code calls into the .cu file with a specified function code, which is
 * used to copy the specified data to the constant by variable name, specific to
 * that .cu file.
 */

#ifndef __MFN_CUDA_CONSTANTCOPYDEFINES_H__
#define __MFN_CUDA_CONSTANTCOPYDEFINES_H__

/**
 * Addresses for the bitmaps stored in constant memory (to be copied to
 * shared memory).  These are typically 4kb, 8kb, or 16kb in size.
 */
#define MFN_CUDA_CONSTANT_BITMAP_A 0x1000
#define MFN_CUDA_CONSTANT_BITMAP_B 0x1001
#define MFN_CUDA_CONSTANT_BITMAP_C 0x1002
#define MFN_CUDA_CONSTANT_BITMAP_D 0x1003
#define MFN_CUDA_CONSTANT_BITMAP_E 0x1004
#define MFN_CUDA_CONSTANT_BITMAP_F 0x1005
#define MFN_CUDA_CONSTANT_BITMAP_G 0x1006
#define MFN_CUDA_CONSTANT_BITMAP_H 0x1007

/**
 * If variable bitmap sizes are in use, altering the mask value for the lookup
 * is required.  This is used to set that mask.
 */
#define MFN_CUDA_CONSTANT_BITMAP_MASK 0x1100

/**
 * The number of hashes currently loaded into the global hashlist in device
 * memory.  This is used for walking the pages after a bitmap hit.
 */
#define MFN_CUDA_NUMBER_OF_HASHES 0x2000

/**
 * The base address of the global hashlist array in device memory.
 */
#define MFN_CUDA_GLOBAL_HASHLIST_ADDRESS 0x3000

/**
 * Addresses for the 128mb (typically) global bitmaps.
 */
#define MFN_CUDA_GLOBAL_BITMAP_A 0x4000
#define MFN_CUDA_GLOBAL_BITMAP_B 0x4001
#define MFN_CUDA_GLOBAL_BITMAP_C 0x4002
#define MFN_CUDA_GLOBAL_BITMAP_D 0x4003
#define MFN_CUDA_GLOBAL_BITMAP_E 0x4004
#define MFN_CUDA_GLOBAL_BITMAP_F 0x4005
#define MFN_CUDA_GLOBAL_BITMAP_G 0x4006
#define MFN_CUDA_GLOBAL_BITMAP_H 0x4007

/**
 * If variable size bitmaps are in use, the bitmap mask for the current size.
 */
#define MFN_CUDA_GLOBAL_BITMAP_MASK 0x4100

/**
 * Addresses for the 256kb cached bitmaps in global memory (to avoid a fully
 * uncached main memory access).
 */
#define MFN_CUDA_GLOBAL_BITMAP_256KB_A 0x5000
#define MFN_CUDA_GLOBAL_BITMAP_256KB_B 0x5001
#define MFN_CUDA_GLOBAL_BITMAP_256KB_C 0x5002
#define MFN_CUDA_GLOBAL_BITMAP_256KB_D 0x5003
#define MFN_CUDA_GLOBAL_BITMAP_256KB_E 0x5004
#define MFN_CUDA_GLOBAL_BITMAP_256KB_F 0x5005
#define MFN_CUDA_GLOBAL_BITMAP_256KB_G 0x5006
#define MFN_CUDA_GLOBAL_BITMAP_256KB_H 0x5007

/**
 * Addresses for the global found password region and the flag region.  Also,
 * a constant for the field length if needed.
 */
#define MFN_CUDA_GLOBAL_FOUND_PASSWORDS 0x6000
#define MFN_CUDA_GLOBAL_FOUND_PASSWORD_FLAGS 0x6001
#define MFN_CUDA_GLOBAL_FOUND_PASSWORDS_FIELD_LENGTH 0x6002

/**
 * The base addresses for the charset arrays.
 */
#define MFN_CUDA_DEVICE_CHARSET_FORWARD 0x7000
#define MFN_CUDA_DEVICE_CHARSET_REVERSE 0x7001
#define MFN_CUDA_DEVICE_CHARSET_LENGTHS 0x7002

/**
 * Various data blocks - password length, steps to run, etc.
 */
#define MFN_CUDA_DEVICE_PASSWORD_LENGTH 0x8000
#define MFN_CUDA_DEVICE_STEPS_TO_RUN 0x8001
#define MFN_CUDA_DEVICE_NUMBER_THREADS 0x8002
#define MFN_CUDA_DEVICE_START_STEP 0x8003

/**
 * Salt related data
 */
#define MFN_CUDA_DEVICE_SALT_DATA 0x9000
#define MFN_CUDA_DEVICE_SALT_LENGTHS 0x9001
#define MFN_CUDA_DEVICE_NUMBER_SALTS 0x9002
#define MFN_CUDA_DEVICE_STARTING_SALT_OFFSET 0x9003

/**
 * Wordlist related data
 */
#define MFN_CUDA_DEVICE_WORDLIST_DATA 0xA000
#define MFN_CUDA_DEVICE_WORDLIST_LENGTHS 0xA001
#define MFN_CUDA_DEVICE_NUMBER_WORDS 0xA002
#define MFN_CUDA_DEVICE_BLOCKS_PER_WORD 0xA003

/**
 * Start points/passwords
 */
#define MFN_CUDA_DEVICE_START_POINTS 0xB000
#define MFN_CUDA_DEVICE_START_PASSWORDS 0xB001


#endif
