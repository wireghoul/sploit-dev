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

#include "Multiforcer_CUDA_host/CHHashTypePlainMD5ofSHA1.h"


// Call the constructor of CHHashTypePlain with len 16
CHHashTypePlainMD5OfSHA1::CHHashTypePlainMD5OfSHA1() : CHHashTypePlain(16) {

}


void CHHashTypePlainMD5OfSHA1::copyDataToConstant(char *hostCharset, int charsetLength,
        unsigned char *hostCharsetLengths, unsigned char *hostSharedBitmap, int threadId) {
    copyMD5OfSHA1DataToConstant(hostCharset, charsetLength, hostCharsetLengths, hostSharedBitmap, threadId);
}

void CHHashTypePlainMD5OfSHA1::Launch_CUDA_Kernel(int passlength, uint64_t charsetLength, int numberOfPasswords, unsigned char *DEVICE_Passwords,
            unsigned char *DEVICE_Success, struct start_positions *DEVICE_Start_Positions,
        uint64_t per_step, uint64_t threads, uint64_t blocks, unsigned char *DEVICE_Hashes,
        unsigned char *DEVICE_Bitmap) {
    // Pass through to the device file
    Launch_CUDA_MD5OfSHA1_Kernel(passlength, charsetLength, numberOfPasswords,
                DEVICE_Passwords, DEVICE_Success,
                DEVICE_Start_Positions, per_step,
                threads, blocks, DEVICE_Hashes,
                DEVICE_Bitmap);
}

