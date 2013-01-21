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
 * This file provides common OpenCL functionality for the kernels.  Things like
 * vector width, vload/vstore types, shared bitmap masks, etc - all in here.
 * 
 * This file does NOT contain functions, hash algorithms, etc.  Those are
 * contained in their own files.  This is just totally commmon stuff.
 */

// Make my UI sane...
#ifndef __OPENCL_VERSION__
    #define __kernel
    #define __global
    #define __local
    #define __private
    #define __constant
    #define get_global_id(x)
    #define get_local_id(x)
    #define restrict
    #include <vector_types.h>
#endif

// Default vector width, if not defined.
#ifndef VECTOR_WIDTH
    #define VECTOR_WIDTH 4
#endif

// Just so this is defined for the GUI tools...
#ifndef PASSWORD_LENGTH
    #define PASSWORD_LENGTH 5
#endif

// Shared bitmap size, in kb.
#ifndef SHARED_BITMAP_SIZE
    #define SHARED_BITMAP_SIZE 8
#endif

// Max supported wordlist length for now
#define MFN_HASH_TYPE_WORDLIST_MAX_PASSLEN 128


/**
 * Define vector-width style constants to use throughout the code.  This enables
 * variable vector width kernels with a minimum of rework.
 * 
 * vector_type : Legacy name for a 32-bit vector type
 * vector_type_8 : Vector of uchar types (unsigned char)
 * vector_type_16 : Vector of ushort types
 * vector_type_32 : Vector of uint types
 * vector_type_64 : Vector of ulong types
 * vload_type : The proper vload function for loading this vector width.
 * vstore_type : The proper vstore function for storing this vector width.
 * convert_type : Convert to a uint vector.
 * convert_type_n : Convert to a vector of the specified width/length.
 * grt_vector_n : A define set to the width.  Used elsewhere in code.
 */
#if VECTOR_WIDTH == 1
    #define vector_type uint
    #define vector_type_8 uchar
    #define vector_type_16 ushort
    #define vector_type_32 uint
    #define vector_type_64 ulong
    #define vload_type vload1
    #define vstore_type vstore1
    #define convert_type
    #define convert_type_8
    #define convert_type_16
    #define convert_type_32
    #define convert_type_64
    // Helpers to emulate vload/vstore for scalars
    #define vload1(offset, p) *(offset + p) 
    #define vstore1(val, offset, p) *(offset + p) = val 
    #define grt_vector_1 1
#elif VECTOR_WIDTH == 2
    #define vector_type uint2
    #define vector_type_8 uchar2
    #define vector_type_16 ushort2
    #define vector_type_32 uint2
    #define vector_type_64 ulong2
    #define vload_type vload2
    #define vstore_type vstore2
    #define convert_type convert_uint2
    #define convert_type_8 convert_uchar2
    #define convert_type_16 convert_ushort2
    #define convert_type_32 convert_uint2
    #define convert_type_64 convert_ulong2
    #define grt_vector_2 1
#elif VECTOR_WIDTH == 4
    #define vector_type uint4
    #define vector_type_8 uchar4
    #define vector_type_16 ushort4
    #define vector_type_32 uint4
    #define vector_type_64 ulong4
    #define vload_type vload4
    #define vstore_type vstore4
    #define convert_type convert_uint4
    #define convert_type_8 convert_uchar4
    #define convert_type_16 convert_ushort4
    #define convert_type_32 convert_uint4
    #define convert_type_64 convert_ulong4
    #define grt_vector_4 1
#elif VECTOR_WIDTH == 8
    #define vector_type uint8
    #define vector_type_8 uchar8
    #define vector_type_16 ushort8
    #define vector_type_32 uint8
    #define vector_type_64 ulong8
    #define vload_type vload8
    #define vstore_type vstore8
    #define convert_type convert_uint8
    #define convert_type_8 convert_uchar8
    #define convert_type_16 convert_ushort8
    #define convert_type_32 convert_uint8
    #define convert_type_64 convert_ulong8
    #define grt_vector_8 1
#elif VECTOR_WIDTH == 16
    #define vector_type uint16
    #define vector_type_8 uchar16
    #define vector_type_16 ushort16
    #define vector_type_32 uint16
    #define vector_type_64 ulong16
    #define vload_type vload16
    #define vstore_type vstore16
    #define convert_type convert_uint16
    #define convert_type_8 convert_uchar16
    #define convert_type_16 convert_ushort16
    #define convert_type_32 convert_uint16
    #define convert_type_64 convert_ulong16
    #define grt_vector_16 1
#else
    #error "Vector width not specified or invalid vector width specified!"
#endif

/**
 * Masks for the shared bitmap.  This is used for the shared/local memory check
 * and should be hardcoded for optimum speed.  SHARED_BITMAP_SIZE is specified
 * in kb.
 */
#if SHARED_BITMAP_SIZE == 2
    #define SHARED_BITMAP_MASK 0x00003fff
#elif SHARED_BITMAP_SIZE == 4
    #define SHARED_BITMAP_MASK 0x00007fff
#elif SHARED_BITMAP_SIZE == 8
    #define SHARED_BITMAP_MASK 0x0000ffff
#elif SHARED_BITMAP_SIZE == 16
    #define SHARED_BITMAP_MASK 0x0001ffff
#elif SHARED_BITMAP_SIZE == 32
    #define SHARED_BITMAP_MASK 0x0003ffff
#endif

// If CPU debug is being used, enable printfs.
#if CPU_DEBUG
    #pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

