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

// This file contains CUDA load/store functions

#ifndef __CUDA_REGISTER_FUNCTIONS_H_
#define __CUDA_REGISTER_FUNCTIONS_H_



// Load data from the global array into the MD5 registers
__device__ inline void SaveMD5RegistersIntoGlobalMemory(uint32_t &b0, uint32_t &b1, uint32_t &b2,
        uint32_t &b3, uint32_t &b4, uint32_t &b5, uint32_t &b6, uint32_t &b7, uint32_t &b8, uint32_t &b9,
        uint32_t &b10, uint32_t &b11, uint32_t &b12, uint32_t &b13, uint32_t &b14, uint32_t &b15,
        uint32_t *InitialArray32, uint32_t &Device_Number_Of_Chains, uint32_t &password_index,
        int password_length) {

    // Load the registers into the needed registers.  Return when done.
    InitialArray32[0 * Device_Number_Of_Chains + password_index] = b0;
    if (password_length <= 4) {
        return;
    }
    InitialArray32[1 * Device_Number_Of_Chains + password_index] = b1;
    if (password_length <= 8) {
        return;
    }
    InitialArray32[2 * Device_Number_Of_Chains + password_index] = b2;
    if (password_length <= 12) {
        return;
    }
    InitialArray32[3 * Device_Number_Of_Chains + password_index] = b3;
    if (password_length <= 16) {
        return;
    }
    InitialArray32[4 * Device_Number_Of_Chains + password_index] = b4;
    if (password_length <= 20) {
        return;
    }
    InitialArray32[5 * Device_Number_Of_Chains + password_index] = b5;
    if (password_length <= 24) {
        return;
    }
    InitialArray32[6 * Device_Number_Of_Chains + password_index] = b6;
    if (password_length <= 28) {
        return;
    }
    InitialArray32[7 * Device_Number_Of_Chains + password_index] = b7;
    if (password_length <= 32) {
        return;
    }
    InitialArray32[8 * Device_Number_Of_Chains + password_index] = b8;
    if (password_length <= 36) {
        return;
    }
    InitialArray32[9 * Device_Number_Of_Chains + password_index] = b9;
    if (password_length <= 40) {
        return;
    }
    InitialArray32[10 * Device_Number_Of_Chains + password_index] = b10;
    if (password_length <= 44) {
        return;
    }
    InitialArray32[11 * Device_Number_Of_Chains + password_index] = b11;
    if (password_length <= 48) {
        return;
    }
}

__device__ inline void LoadMD5RegistersFromGlobalMemory(uint32_t &b0, uint32_t &b1, uint32_t &b2,
        uint32_t &b3, uint32_t &b4, uint32_t &b5, uint32_t &b6, uint32_t &b7, uint32_t &b8, uint32_t &b9,
        uint32_t &b10, uint32_t &b11, uint32_t &b12, uint32_t &b13, uint32_t &b14, uint32_t &b15,
        uint32_t *InitialArray32, uint32_t &Device_Number_Of_Chains, uint32_t &password_index,
        int password_length) {

    // Load the registers into the needed registers.  Return when done.
    b0 = (uint32_t)InitialArray32[0 * Device_Number_Of_Chains + password_index];
    if (password_length <= 4) {
        return;
    }
    b1 = (uint32_t)InitialArray32[1 * Device_Number_Of_Chains + password_index];
    if (password_length <= 8) {
        return;
    }
    b2 = (uint32_t)InitialArray32[2 * Device_Number_Of_Chains + password_index];
    if (password_length <= 12) {
        return;
    }
    b3 = (uint32_t)InitialArray32[3 * Device_Number_Of_Chains + password_index];
    if (password_length <= 16) {
        return;
    }
    b4 = (uint32_t)InitialArray32[4 * Device_Number_Of_Chains + password_index];
    if (password_length <= 20) {
        return;
    }
    b5 = (uint32_t)InitialArray32[5 * Device_Number_Of_Chains + password_index];
    if (password_length <= 24) {
        return;
    }
    b6 = (uint32_t)InitialArray32[6 * Device_Number_Of_Chains + password_index];
    if (password_length <= 28) {
        return;
    }
    b7 = (uint32_t)InitialArray32[7 * Device_Number_Of_Chains + password_index];
    if (password_length <= 32) {
        return;
    }
    b8 = (uint32_t)InitialArray32[8 * Device_Number_Of_Chains + password_index];
    if (password_length <= 36) {
        return;
    }
    b9 = (uint32_t)InitialArray32[9 * Device_Number_Of_Chains + password_index];
    if (password_length <= 40) {
        return;
    }
    b10 = (uint32_t)InitialArray32[10 * Device_Number_Of_Chains + password_index];
    if (password_length <= 44) {
        return;
    }
    b11 = (uint32_t)InitialArray32[11 * Device_Number_Of_Chains + password_index];
    if (password_length <= 48) {
        return;
    }
}

__device__ inline void SaveNTLMRegistersIntoGlobalMemory(uint32_t &b0, uint32_t &b1, uint32_t &b2,
        uint32_t &b3, uint32_t &b4, uint32_t &b5, uint32_t &b6, uint32_t &b7, uint32_t &b8, uint32_t &b9,
        uint32_t &b10, uint32_t &b11, uint32_t &b12, uint32_t &b13, uint32_t &b14, uint32_t &b15,
        uint32_t *InitialArray32, uint32_t &Device_Number_Of_Chains, uint32_t &password_index,
        int password_length) {

    // This *should* work even if there's a padding bit - it will just be set
    // again on the next iteration.

    b15 =  (b0 & 0xff) | ((b0 & 0xff0000) >> 8);
    if (password_length <= 2) {
        InitialArray32[0 * Device_Number_Of_Chains + password_index] = b15;
        return;
    }

    b15 |= (b1 & 0xff) << 16 | ((b1 & 0xff0000) << 8);
    InitialArray32[0 * Device_Number_Of_Chains + password_index] = b15;

    if (password_length <= 4) {
        return;
    }

    b15 =  (b2 & 0xff) | ((b2 & 0xff0000) >> 8);
    if (password_length <= 6) {
        InitialArray32[1 * Device_Number_Of_Chains + password_index] = b15;
        return;
    }

    b15 |= (b3 & 0xff) << 16 | ((b3 & 0xff0000) << 8);
    InitialArray32[1 * Device_Number_Of_Chains + password_index] = b15;

    if (password_length <= 8) {
        return;
    }

    b15 =  (b4 & 0xff) | ((b4 & 0xff0000) >> 8);
    if (password_length <= 10) {
        InitialArray32[2 * Device_Number_Of_Chains + password_index] = b15;
        return;
    }

    b15 |= (b5 & 0xff) << 16 | ((b5 & 0xff0000) << 8);
    InitialArray32[2 * Device_Number_Of_Chains + password_index] = b15;

    if (password_length <= 12) {
        return;
    }

}

// Relies on the unused slots being zeroed.
__device__ inline void LoadNTLMRegistersFromGlobalMemory(uint32_t &b0, uint32_t &b1, uint32_t &b2,
        uint32_t &b3, uint32_t &b4, uint32_t &b5, uint32_t &b6, uint32_t &b7, uint32_t &b8, uint32_t &b9,
        uint32_t &b10, uint32_t &b11, uint32_t &b12, uint32_t &b13, uint32_t &b14, uint32_t &b15,
        uint32_t *InitialArray32, uint32_t &Device_Number_Of_Chains, uint32_t &password_index,
        int password_length) {

    // General flow: Load data into b15, split it up as needed, return when done.
    // This works in bits of two - makes it easier to code, no real speed loss.
    // As the initial array is zeroed, this will work just fine and put null bytes
    // where there is no other data.

    // First uint32_t...
    b15 = (uint32_t)InitialArray32[0 * Device_Number_Of_Chains + password_index];
    b0 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
    if (password_length <= 2) {
        b15 = 0x00000000;
        return;
    }

    b1 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
    if (password_length <= 4) {
        b15 = 0x00000000;
        return;
    }

    // Second uint32_t...
    b15 = (uint32_t)InitialArray32[1 * Device_Number_Of_Chains + password_index];
    b2 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
    if (password_length <= 6) {
        b15 = 0x00000000;
        return;
    }

    b3 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
    if (password_length <= 8) {
        b15 = 0x00000000;
        return;
    }

    // Third uint32_t...
    b15 = (uint32_t)InitialArray32[2 * Device_Number_Of_Chains + password_index];
    b4 = (b15 & 0xff) | ((b15 & 0xff00) << 8);
    if (password_length <= 6) {
        b15 = 0x00000000;
        return;
    }

    b5 = ((b15 & 0xff0000) >> 16) | ((b15 & 0xff000000) >> 8);
    if (password_length <= 8) {
        b15 = 0x00000000;
        return;
    }
}


#endif