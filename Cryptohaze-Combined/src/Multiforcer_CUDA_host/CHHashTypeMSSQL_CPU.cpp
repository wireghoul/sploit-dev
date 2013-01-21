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

#include "Multiforcer_CUDA_host/CHHashTypeMSSQL_CPU.h"


inline void clearB0toB15(UINT4 &b0, UINT4 &b1, UINT4 &b2, UINT4 &b3, UINT4 &b4, UINT4 &b5, UINT4 &b6, UINT4 &b7,
               UINT4 &b8, UINT4 &b9, UINT4 &b10, UINT4 &b11, UINT4 &b12, UINT4 &b13, UINT4 &b14, UINT4 &b15) {
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

// Init MSSQL functions.
inline void CPUinitMSSQL(char *password, int length, uint32_t salt,
        UINT4 &b0, UINT4 &b1, UINT4 &b2, UINT4 &b3, UINT4 &b4, UINT4 &b5, UINT4 &b6, UINT4 &b7,
        UINT4 &b8, UINT4 &b9, UINT4 &b10, UINT4 &b11, UINT4 &b12, UINT4 &b13, UINT4 &b14, UINT4 &b15) {

  uint32_t salt_original;
  clearB0toB15(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);

  // Swap the salt yet again...
  salt_original = salt;
  salt = ((salt_original >> 24) & 0xff) | (((salt_original >> 16) & 0xff) << 8) |
    (((salt_original >> 8) & 0xff) << 16) | (((salt_original) & 0xff) << 24);

  // Set length properly (length in bits)
  // Note that this is /different/ from MD!
  // Add 4 bytes for the salt!
  b15 = ((length * 2) + 4) * 8 << 24;

 if (length == 0) {
    b0 = salt;
    b1 |= 0x00000080;
    return;
  }
  b0 |= password[0];
  if (length == 1) {
    // Set the salt byte
    b0 &= 0x0000ffff;
    b0 |= (salt & 0xffff) << 16;
    b1 &= 0xffff0000;
    b1 |= ((salt & 0xffff0000) >> 16);

    b1 |= 0x00800000;
    return;
  }
  b0 |= password[1] << 16;
  if (length == 2) {
    b1 = salt;
    b2 |= 0x00000080;
	return;
  }
  b1 |= password[2];
  if (length == 3) {
    // Set the salt byte
    // Mask off the high bits - we have the low bits already.
    b1 &= 0x0000ffff;
    b1 |= (salt & 0xffff) << 16;
    // Mask off the high bits of b2
    b2 &= 0xffff0000;
    b2 |= ((salt & 0xffff0000) >> 16);
    b2 |= 0x00800000;
    return;
  }
  b1 |= password[3] << 16;
  if (length == 4) {
    b2 = salt;
    b3 |= 0x00000080;
    return;
  }
  b2 |= password[4];
  if (length == 5) {
    b2 &= 0x0000ffff;
    b2 |= ((salt) & 0xffff) << 16;
    b3 &= 0xffff0000;
    b3 |= ((salt & 0xffff0000) >> 16);
    b3 |= 0x00800000;
    return;
  }
  b2 |= password[5] << 16;
  if (length == 6) {
    b3 = salt;
    b4 |= 0x00000080;
    return;
  }
  b3 |= password[6];
  if (length == 7) {
    b3 &= 0x0000ffff;
    b3 |= ((salt) & 0xffff) << 16;
    b4 &= 0xffff0000;
    b4 |= ((salt & 0xffff0000) >> 16);
    b4 |= 0x00800000;
    return;
  }
  b3 |= password[7] << 16;
  if (length == 8) {
    b4 = salt;
    b5 |= 0x00000080;
    return;
  }
  b4 |= password[8];
  if (length == 9) {
    b4 &= 0x0000ffff;
    b4 |= ((salt) & 0xffff) << 16;
    b5 &= 0xffff0000;
    b5 |= ((salt & 0xffff0000) >> 16);
    b5 |= 0x00800000;
    return;
  }
  b4 |= password[9] << 16;
  if (length == 10) {
    b5 = salt;
    b6 |= 0x00000080;
    return;
  }
  b5 |= password[10];
  if (length == 11) {
    b5 &= 0x0000ffff;
    b5 |= ((salt) & 0xffff) << 16;
    b6 &= 0xffff0000;
    b7 |= ((salt & 0xffff0000) >> 16);
    b6 |= 0x00800000;
    return;
  }
  b5 |= password[11] << 16;
  if (length == 12) {
    b6 = salt;
    b7 |= 0x00000080;
    return;
  }
  b6 |= password[12];
  if (length == 13) {
    b6 &= 0x0000ffff;
    b7 |= ((salt) & 0xffff) << 16;
    b7 &= 0xffff0000;
    b7 |= ((salt & 0xffff0000) >> 16);
    b7 |= 0x00800000;
    return;
  }
  b6 |= password[13] << 16;
  if (length == 14) {
    b7 = salt;
    b8 |= 0x00000080;
    return;
  }
  b7 |= password[14];
  if (length == 15) {
    b7 &= 0x0000ffff;
    b7 |= ((salt) & 0xffff) << 16;
    b8 &= 0xffff0000;
    b8 |= ((salt & 0xffff0000) >> 16);
    b8 |= 0x00800000;
    return;
  }
  b7 |= password[15] << 16;
  if (length == 16) {
    b8 = salt;
    b9 |= 0x00000080;
    return;
  }
}


// Returns 0 on failure, 1 on match
int CPU_MSSQL(char *password, uint32_t salt, unsigned char *targetHash) {
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15; \
    uint32_t a,b,c,d,e;
    int length;

    uint32_t *targetHash32 = (uint32_t *)targetHash;

    length = strlen((const char *)password);

    CPUinitMSSQL(password, length, salt,
        b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);

    SHA_TRANSFORM(a, b, c, d, e, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);

    a = reverse(a);
    b = reverse(b);
    c = reverse(c);
    d = reverse(d);
    e = reverse(e);

    if (a == targetHash32[0]) {
        if (b == targetHash32[1]) {
            if (c == targetHash32[2]) {
                if (d == targetHash32[3]) {
                    if (e == targetHash32[4]) {
                        return 1;
                    }
                }
            }
        }
    }
    return 0;
}