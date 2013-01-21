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

// This file contains CUDA reduction functions

#ifndef __CUDA_REDUCTION_FUNCTIONS_H_
#define __CUDA_REDUCTION_FUNCTIONS_H_

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

__device__ inline void reduceSingleCharsetNormal(uint32_t &b0, uint32_t &b1, uint32_t &b2,
        uint32_t a, uint32_t b, uint32_t c, uint32_t d,
        uint32_t CurrentStep, char charset[], uint32_t charset_offset, int PasswordLength, uint32_t Device_Table_Index) {

    uint32_t z;
    // Reduce it
    // First 3
    z = (uint32_t)(a+CurrentStep+Device_Table_Index) % (256*256*256);
    b0 = (uint32_t)charset[(z % 256) + charset_offset];
    if (PasswordLength == 1) {return;}
    z /= 256;
    b0 |= (uint32_t)charset[(z % 256) + charset_offset] << 8;
    if (PasswordLength == 2) {return;}
    z /= 256;
    b0 |= (uint32_t)charset[(z % 256) + charset_offset] << 16;
    if (PasswordLength == 3) {return;}

    // Second 3
    z = (uint32_t)(b+CurrentStep+Device_Table_Index) % (256*256*256);
    b0 |= (uint32_t)charset[(z % 256) + charset_offset] << 24;
    if (PasswordLength == 4) {return;}
    z /= 256;
    b1 = (uint32_t)charset[(z % 256) + charset_offset];
    if (PasswordLength == 5) {return;}
    z /= 256;
    b1 |= (uint32_t)charset[(z % 256) + charset_offset] << 8;
    if (PasswordLength == 6) {return;}

    // Last 2
    z = (uint32_t)(c+CurrentStep+Device_Table_Index) % (256*256*256);
    b1 |= (uint32_t)charset[(z % 256) + charset_offset] << 16;
    if (PasswordLength == 7) {return;}
    z /= 256;
    b1 |= (uint32_t)charset[(z % 256) + charset_offset] << 24;
    if (PasswordLength == 8) {return;}
    z /= 256;
    b2 = (uint32_t)charset[(z % 256) + charset_offset];
    if (PasswordLength == 9) {return;}

    z = (uint32_t)(d+CurrentStep+Device_Table_Index) % (256*256*256);
    b2 |= (uint32_t)charset[(z % 256) + charset_offset] << 8;
    if (PasswordLength == 10) {return;}
    z /= 256;
    b2 |= (uint32_t)charset[(z % 256) + charset_offset] << 16;
    if (PasswordLength == 11) {return;}
    z /= 256;
    b2 |= (uint32_t)charset[(z % 256) + charset_offset] << 24;
    if (PasswordLength == 12) {return;}

}


// For NTLM use
__device__ inline void reduceSingleCharsetNTLM(uint32_t &b0, uint32_t &b1, uint32_t &b2,
        uint32_t &b3, uint32_t &b4, uint32_t a, uint32_t b, uint32_t c, uint32_t d,
        uint32_t CurrentStep, char charset[], uint32_t charset_offset, int PasswordLength, uint32_t Device_Table_Index) {

    uint32_t z;
    // Reduce it
    // First 3
    z = (uint32_t)(a+CurrentStep+Device_Table_Index) % (256*256*256);
    b0 = (uint32_t)charset[(z % 256) + charset_offset];
    if (PasswordLength == 1) {return;}
    z /= 256;
    b0 |= (uint32_t)charset[(z % 256) + charset_offset] << 16;
    if (PasswordLength == 2) {return;}
    z /= 256;
    b1 = (uint32_t)charset[(z % 256) + charset_offset];
    if (PasswordLength == 3) {return;}

    // Second 3
    z = (uint32_t)(b+CurrentStep+Device_Table_Index) % (256*256*256);
    b1 |= (uint32_t)charset[(z % 256) + charset_offset] << 16;
    if (PasswordLength == 4) {return;}
    z /= 256;
    b2 = (uint32_t)charset[(z % 256) + charset_offset];
    if (PasswordLength == 5) {return;}
    z /= 256;
    b2 |= (uint32_t)charset[(z % 256) + charset_offset] << 16;
    if (PasswordLength == 6) {return;}

    // Last 2
    z = (uint32_t)(c+CurrentStep+Device_Table_Index) % (256*256*256);
    b3 = (uint32_t)charset[(z % 256) + charset_offset];
    if (PasswordLength == 7) {return;}
    z /= 256;
    b3 |= (uint32_t)charset[(z % 256) + charset_offset] << 16;
    if (PasswordLength == 8) {return;}
    z /= 256;
    b4 = (uint32_t)charset[(z % 256) + charset_offset];
    if (PasswordLength == 9) {return;}

    z = (uint32_t)(d+CurrentStep+Device_Table_Index) % (256*256*256);
    b4 |= (uint32_t)charset[(z % 256) + charset_offset] << 16;
    if (PasswordLength == 10) {return;}
    /*
    z /= 256;
    b5 |= (uint32_t)charset[(z % 256) + charset_offset] << 16;
    if (PasswordLength == 11) {return;}
    z /= 256;
    b5 |= (uint32_t)charset[(z % 256) + charset_offset] << 24;
    if (PasswordLength == 12) {return;}
    // // Only supporting through 10 right now
    */

}

#endif