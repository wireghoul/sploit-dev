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
 * This class is a standalone writer for metaprograms needed by the OpenCL
 * kernels.  It is generally going to create a string that contains a define
 * which the kernels call.  These defines are written at runtime, right before
 * compilation, and include data about things like password length and the
 * vector width of the kernels.  This improves performance a whole heck of a 
 * lot over static defines, and is worth the additional uglyness.
 * 
 * Note: This should be called by the kernel-specific class, so that it can call
 * the proper function, as it may vary from kernel type to kernel type.
 */

#ifndef __MFNOPENCLMETAPROGRAMS_H
#define __MFNOPENCLMETAPROGRAMS_H

#include <string>
#include <vector>
#include <stdint.h>


class MFNOpenCLMetaprograms {
public:
    /**
     * Makes a password incrementor define for a normal MD5 type hash function.
     * 
     * This function makes a password incrementor define for a little endian
     * hash function with "packed" passwords (1 byte per character) such as MD5.
     * 
     * It outputs a define that will handle the vector password incrementing for
     * the specified parameters
     * 
     * #define name: OpenCLPasswordIncrementorLE
     * 
     * @param passwordLength The length of the password
     * @param vectorWidth The vector width of the kernel (2, 4, 8, 16)
     * @return A string containing the define to be included in the compile.
     */
    std::string makePasswordSingleIncrementorsLE(int passwordLength, int vectorWidth);

    std::string makePasswordSingleIncrementorsBE(int passwordLength, int vectorWidth);

    std::string makePasswordSingleIncrementorsNTLM(int passwordLength, int vectorWidth);
    
    /**
     * Makes a memory-less password incrementor define for a little endian hash
     * 
     * Extension of the above.
     * 
     * @param passwordLength The length of the password being cracked.
     * @param vectorWidth The vector width of the kernel.
     * @param charset The current charset vector (only element 0 will be used).
     * @param passStride Set to 1 for MD5, 2 for NTLM.
     * @param reverseMacroStep The step to run the REVERSE() macro - single hash.
     * @return A string containing the new charset incrementor.
     */
    std::string makePasswordNoMemSingleIncrementorsLE(int passwordLength, 
        int vectorWidth, std::vector<std::vector<uint8_t> > charset, 
        int passStride = 1, int reverseMacroStep = 0);

    std::string makePasswordNoMemMultipleIncrementorsLE(int passwordLength, 
        int vectorWidth, std::vector<std::vector<uint8_t> > charset,
        int passStride = 1, int reverseMacroStep = 0);

    std::string makePasswordNoMemSingleIncrementorsBE(int passwordLength, 
        int vectorWidth, std::vector<std::vector<uint8_t> > charset, 
        int passStride = 1, int reverseMacroStep = 0);

    std::string makePasswordNoMemMultipleIncrementorsBE(int passwordLength, 
        int vectorWidth, std::vector<std::vector<uint8_t> > charset,
        int passStride = 1, int reverseMacroStep = 0);
    
    
    /**
     * Makes a password incrementor define for a normal MD5 type hash function
     * with per-position charsets in use.
     * 
     * This function makes a password incrementor define for a little endian
     * hash function with "packed" passwords (1 byte per character) such as MD5.
     * 
     * It outputs a define that will handle the vector password incrementing for
     * the specified parameters.
     * 
     * #define name: OpenCLPasswordIncrementorLE
     * 
     * @param passwordLength The length of the password
     * @param vectorWidth The vector width of the kernel (2, 4, 8, 16)
     * @param charsetBufferLength The length of the each position charset buffer on the device.
     * @return A string containing the define to be included in the compile.
     */
    std::string makePasswordMultipleIncrementorsLE(int passwordLength, int vectorWidth,
        int charsetBufferLength);

    std::string makePasswordMultipleIncrementorsBE(int passwordLength, int vectorWidth,
        int charsetBufferLength);

    std::string makePasswordMultipleIncrementorsNTLM(int passwordLength, int vectorWidth,
        int charsetBufferLength);

   
    /**
     * Generate the checker for the filled bitmaps.
     * 
     * This generates a vector-width wide checker for passwords.  It checks them
     * against the assorted bitmaps, then finally calls the check and report
     * function if they pass the bitmaps.
     * 
     * This function will only generate the bitmap lookups if the bitmap is
     * present on the device.  This avoids some unneeded checks.
     * 
     * @param vectorWidth Vector width of the kernel
     * @param numberBigBitmapsFilled Number of "big bitmaps" populated on the GPU
     * @param passwordCheckFunction The name of the password verification/reporting function.
     * @return A string with the needed define.
     */
    std::string makeBitmapLookup(int vectorWidth, int numberBigBitmapsFilled, 
        std::string passwordCheckFunction);
    
    /**
     * This generates a bitmap lookup function that supports "early out" optimizations.
     * 
     * Early out is a set of optimizations that involve checking the first output
     * variable, and if it doesn't match bitmaps, not bothering with the rest
     * of the bitmaps.  If the first variable matches, the other variables are
     * calculated and checked as needed.  This is worth around 5% in performance
     * and is worth the complexity.
     * 
     * For each bitmap, the letter (a, b, c, d) is provided, as well as
     * information as to if the bitmap is present (as not all will always be).
     * 
     * Finally, the calculation string needed to get the next value (if needed)
     * is provided, and will be inserted in the right place.  Terribly complex,
     * and worth 5% of a really big number...
     * 
     * lookupFunctionName specifies the name of the function to create.
     * 
     * ifFoundRunMacro specifies a macro to run before actually starting to do
     * the searches.  This is useful for things that need to load the password
     * back into b0,b1, ... before doing the search because it has been
     * overwritten by other data.  Used, among other places, in the IPB kernel.
     * 
     * @param vectorWidth Vector width of the kernel.
     * @param passwordCheckFunction Name of the password check function.
     * @param bitmap_0_letter
     * @param bitmap_0_is_present
     * @param bitmap_1_letter
     * @param bitmap_1_is_present
     * @param bitmap_1_calculation_string
     * @param bitmap_2_letter
     * @param bitmap_2_is_present
     * @param bitmap_2_calculation_string
     * @param bitmap_3_letter
     * @param bitmap_3_is_present
     * @param bitmap_3_calculation_string
     * @param use_l2_bitmap True if L2 bitmap is to be used.
     * @return 
     */
    
    std::string makeBitmapLookupEarlyOut(int vectorWidth,  std::string passwordCheckFunction,
        char bitmap_0_letter, int bitmap_0_is_present, 
        char bitmap_1_letter, int bitmap_1_is_present, std::string bitmap_1_calculation_string,
        char bitmap_2_letter, int bitmap_2_is_present, std::string bitmap_2_calculation_string,
        char bitmap_3_letter, int bitmap_3_is_present, std::string bitmap_3_calculation_string,
        char use_l2_bitmap = 0,
        std::string lookupFunctionName = std::string("OpenCLPasswordCheck128_EarlyOut"),
        std::string ifFoundRunMacro = std::string(""));
        
    /**
     * Makes a define that copies a specified number of elements from src to dst.
     * 
     * This is useful for unrolling loops, since AMD's #pragma unroll is STILL
     * broken...
     * 
     * #define copierName(src, dst) { ...
     * 
     * @param copierName Name of the copier define.
     * @param numberElements Number of elements to copy.
     * @return 
     */
    std::string makeCopyOperator(std::string copierName, int numberElements);
    
};

#endif
