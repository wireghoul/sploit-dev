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

/* MD4 Defines as per RFC reference implementation */
#define MD4F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define MD4G(x, y, z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
#define MD4H(x, y, z) ((x) ^ (y) ^ (z))
#define MD4ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))
#define MD4FF(a, b, c, d, x, s) { \
    (a) += MD4F ((b), (c), (d)) + (x); \
    (a) = MD4ROTATE_LEFT ((a), (s)); \
  }
#define MD4GG(a, b, c, d, x, s) { \
    (a) += MD4G ((b), (c), (d)) + (x) + (uint32_t)0x5a827999; \
    (a) = MD4ROTATE_LEFT ((a), (s)); \
  }
#define MD4HH(a, b, c, d, x, s) { \
    (a) += MD4H ((b), (c), (d)) + (x) + (uint32_t)0x6ed9eba1; \
    (a) = MD4ROTATE_LEFT ((a), (s)); \
  }
#define MD4S11 3
#define MD4S12 7
#define MD4S13 11
#define MD4S14 19
#define MD4S21 3
#define MD4S22 5
#define MD4S23 9
#define MD4S24 13
#define MD4S31 3
#define MD4S32 9
#define MD4S33 11
#define MD4S34 15
/* End MD4 Defines */



#define MD4_FIRST_2_ROUNDS() { \
a = 0x67452301; \
b = 0xefcdab89; \
c = 0x98badcfe; \
d = 0x10325476; \
  MD4FF (a, b, c, d, b0, MD4S11); \
  MD4FF (d, a, b, c, b1, MD4S12); \
  MD4FF (c, d, a, b, b2, MD4S13); \
  MD4FF (b, c, d, a, b3, MD4S14); \
  MD4FF (a, b, c, d, b4, MD4S11); \
  MD4FF (d, a, b, c, b5, MD4S12); \
  MD4FF (c, d, a, b, b6, MD4S13); \
  MD4FF (b, c, d, a, b7, MD4S14); \
  MD4FF (a, b, c, d, b8, MD4S11); \
  MD4FF (d, a, b, c, b9, MD4S12); \
  MD4FF (c, d, a, b, b10, MD4S13); \
  MD4FF (b, c, d, a, b11, MD4S14); \
  MD4FF (a, b, c, d, b12, MD4S11); \
  MD4FF (d, a, b, c, b13, MD4S12); \
  MD4FF (c, d, a, b, b14, MD4S13); \
  MD4FF (b, c, d, a, b15, MD4S14); \
  MD4GG (a, b, c, d, b0, MD4S21); \
  MD4GG (d, a, b, c, b4, MD4S22); \
  MD4GG (c, d, a, b, b8, MD4S23); \
  MD4GG (b, c, d, a, b12, MD4S24); \
  MD4GG (a, b, c, d, b1, MD4S21); \
  MD4GG (d, a, b, c, b5, MD4S22); \
  MD4GG (c, d, a, b, b9, MD4S23); \
  MD4GG (b, c, d, a, b13, MD4S24); \
  MD4GG (a, b, c, d, b2, MD4S21); \
  MD4GG (d, a, b, c, b6, MD4S22); \
  MD4GG (c, d, a, b, b10, MD4S23); \
  MD4GG (b, c, d, a, b14, MD4S24); \
  MD4GG (a, b, c, d, b3, MD4S21); \
  MD4GG (d, a, b, c, b7, MD4S22); \
  MD4GG (c, d, a, b, b11, MD4S23); \
  MD4GG (b, c, d, a, b15, MD4S24); \
}

#define MD4_FULL_HASH() { \
a = 0x67452301; \
b = 0xefcdab89; \
c = 0x98badcfe; \
d = 0x10325476; \
  MD4FF (a, b, c, d, b0, MD4S11); \
  MD4FF (d, a, b, c, b1, MD4S12); \
  MD4FF (c, d, a, b, b2, MD4S13); \
  MD4FF (b, c, d, a, b3, MD4S14); \
  MD4FF (a, b, c, d, b4, MD4S11); \
  MD4FF (d, a, b, c, b5, MD4S12); \
  MD4FF (c, d, a, b, b6, MD4S13); \
  MD4FF (b, c, d, a, b7, MD4S14); \
  MD4FF (a, b, c, d, b8, MD4S11); \
  MD4FF (d, a, b, c, b9, MD4S12); \
  MD4FF (c, d, a, b, b10, MD4S13); \
  MD4FF (b, c, d, a, b11, MD4S14); \
  MD4FF (a, b, c, d, b12, MD4S11); \
  MD4FF (d, a, b, c, b13, MD4S12); \
  MD4FF (c, d, a, b, b14, MD4S13); \
  MD4FF (b, c, d, a, b15, MD4S14); \
  MD4GG (a, b, c, d, b0, MD4S21); \
  MD4GG (d, a, b, c, b4, MD4S22); \
  MD4GG (c, d, a, b, b8, MD4S23); \
  MD4GG (b, c, d, a, b12, MD4S24); \
  MD4GG (a, b, c, d, b1, MD4S21); \
  MD4GG (d, a, b, c, b5, MD4S22); \
  MD4GG (c, d, a, b, b9, MD4S23); \
  MD4GG (b, c, d, a, b13, MD4S24); \
  MD4GG (a, b, c, d, b2, MD4S21); \
  MD4GG (d, a, b, c, b6, MD4S22); \
  MD4GG (c, d, a, b, b10, MD4S23); \
  MD4GG (b, c, d, a, b14, MD4S24); \
  MD4GG (a, b, c, d, b3, MD4S21); \
  MD4GG (d, a, b, c, b7, MD4S22); \
  MD4GG (c, d, a, b, b11, MD4S23); \
  MD4GG (b, c, d, a, b15, MD4S24); \
  MD4HH (a, b, c, d, b0, MD4S31); \
  MD4HH (d, a, b, c, b8, MD4S32); \
  MD4HH (c, d, a, b, b4, MD4S33); \
  MD4HH (b, c, d, a, b12, MD4S34); \
  MD4HH (a, b, c, d, b2, MD4S31); \
  MD4HH (d, a, b, c, b10, MD4S32); \
  MD4HH (c, d, a, b, b6, MD4S33); \
  MD4HH (b, c, d, a, b14, MD4S34); \
  MD4HH (a, b, c, d, b1, MD4S31); \
  MD4HH (d, a, b, c, b9, MD4S32); \
  MD4HH (c, d, a, b, b5, MD4S33); \
  MD4HH (b, c, d, a, b13, MD4S34); \
  MD4HH (a, b, c, d, b3, MD4S31); \
  MD4HH (d, a, b, c, b11, MD4S32); \
  MD4HH (c, d, a, b, b7, MD4S33); \
  MD4HH (b, c, d, a, b15, MD4S34); \
  a += 0x67452301; \
  b += 0xefcdab89; \
  c += 0x98badcfe; \
  d += 0x10325476; \
}
