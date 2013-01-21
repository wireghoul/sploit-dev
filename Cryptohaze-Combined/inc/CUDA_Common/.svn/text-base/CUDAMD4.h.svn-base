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
    (a) += MD4G ((b), (c), (d)) + (x) + (UINT4)0x5a827999; \
    (a) = MD4ROTATE_LEFT ((a), (s)); \
  }
#define MD4HH(a, b, c, d, x, s) { \
    (a) += MD4H ((b), (c), (d)) + (x) + (UINT4)0x6ed9eba1; \
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

/*
 * MD4 code: Call as:
 * MD4(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d);
 * Takes inputs from b0-b15 and returns a, b, c, d.
*/
__device__ inline void CUDA_MD4(UINT4 b0, UINT4 b1, UINT4 b2, UINT4 b3, UINT4 b4, UINT4 b5, UINT4 b6, UINT4 b7,
			   UINT4 b8, UINT4 b9, UINT4 b10, UINT4 b11, UINT4 b12, UINT4 b13, UINT4 b14, UINT4 b15,
			   UINT4 &a, UINT4 &b, UINT4 &c, UINT4 &d) {
  a = 0x67452301;
  b = 0xefcdab89;
  c = 0x98badcfe;
  d = 0x10325476;

  MD4FF (a, b, c, d, b0, MD4S11); /* 1 */
  MD4FF (d, a, b, c, b1, MD4S12); /* 2 */
  MD4FF (c, d, a, b, b2, MD4S13); /* 3 */
  MD4FF (b, c, d, a, b3, MD4S14); /* 4 */
  MD4FF (a, b, c, d, b4, MD4S11); /* 5 */
  MD4FF (d, a, b, c, b5, MD4S12); /* 6 */
  MD4FF (c, d, a, b, b6, MD4S13); /* 7 */
  MD4FF (b, c, d, a, b7, MD4S14); /* 8 */
  MD4FF (a, b, c, d, b8, MD4S11); /* 9 */
  MD4FF (d, a, b, c, b9, MD4S12); /* 10 */
  MD4FF (c, d, a, b, b10, MD4S13); /* 11 */
  MD4FF (b, c, d, a, b11, MD4S14); /* 12 */
  MD4FF (a, b, c, d, b12, MD4S11); /* 13 */
  MD4FF (d, a, b, c, b13, MD4S12); /* 14 */
  MD4FF (c, d, a, b, b14, MD4S13); /* 15 */
  MD4FF (b, c, d, a, b15, MD4S14); /* 16 */

  /* Round 2 */
  MD4GG (a, b, c, d, b0, MD4S21); /* 17 */
  MD4GG (d, a, b, c, b4, MD4S22); /* 18 */
  MD4GG (c, d, a, b, b8, MD4S23); /* 19 */
  MD4GG (b, c, d, a, b12, MD4S24); /* 20 */
  MD4GG (a, b, c, d, b1, MD4S21); /* 21 */
  MD4GG (d, a, b, c, b5, MD4S22); /* 22 */
  MD4GG (c, d, a, b, b9, MD4S23); /* 23 */
  MD4GG (b, c, d, a, b13, MD4S24); /* 24 */
  MD4GG (a, b, c, d, b2, MD4S21); /* 25 */
  MD4GG (d, a, b, c, b6, MD4S22); /* 26 */
  MD4GG (c, d, a, b, b10, MD4S23); /* 27 */
  MD4GG (b, c, d, a, b14, MD4S24); /* 28 */
  MD4GG (a, b, c, d, b3, MD4S21); /* 29 */
  MD4GG (d, a, b, c, b7, MD4S22); /* 30 */
  MD4GG (c, d, a, b, b11, MD4S23); /* 31 */
  MD4GG (b, c, d, a, b15, MD4S24); /* 32 */


  /* Round 3 */
  MD4HH (a, b, c, d, b0, MD4S31); /* 33 */
  MD4HH (d, a, b, c, b8, MD4S32); /* 34 */
  MD4HH (c, d, a, b, b4, MD4S33); /* 35 */
  MD4HH (b, c, d, a, b12, MD4S34); /* 36 */
  MD4HH (a, b, c, d, b2, MD4S31); /* 37 */
  MD4HH (d, a, b, c, b10, MD4S32); /* 38 */
  MD4HH (c, d, a, b, b6, MD4S33); /* 39 */
  MD4HH (b, c, d, a, b14, MD4S34); /* 40 */
  MD4HH (a, b, c, d, b1, MD4S31); /* 41 */
  MD4HH (d, a, b, c, b9, MD4S32); /* 42 */
  MD4HH (c, d, a, b, b5, MD4S33); /* 43 */
  MD4HH (b, c, d, a, b13, MD4S34); /* 44 */
  MD4HH (a, b, c, d, b3, MD4S31); /* 45 */
  MD4HH (d, a, b, c, b11, MD4S32); /* 46 */
  MD4HH (c, d, a, b, b7, MD4S33); /* 47 */
  MD4HH (b, c, d, a, b15, MD4S34); /* 48 */

  // Finally, add initial values, as this is the only pass we make.
  a += 0x67452301;
  b += 0xefcdab89;
  c += 0x98badcfe;
  d += 0x10325476;
}



__device__ inline void CUDA_GENERIC_MD4(UINT4 b0, UINT4 b1, UINT4 b2, UINT4 b3, UINT4 b4, UINT4 b5, UINT4 b6, UINT4 b7,
			   UINT4 b8, UINT4 b9, UINT4 b10, UINT4 b11, UINT4 b12, UINT4 b13, UINT4 b14, UINT4 b15,
			   UINT4 &a, UINT4 &b, UINT4 &c, UINT4 &d,
                           int data_length_bytes) {

  // Set length properly (length in bits)
  b14 = data_length_bytes * 8;

  // Set the padding byte
  SetCharacterAtPosition(0x80, data_length_bytes,
        b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15 );

  a = 0x67452301;
  b = 0xefcdab89;
  c = 0x98badcfe;
  d = 0x10325476;

  MD4FF (a, b, c, d, b0, MD4S11); /* 1 */
  MD4FF (d, a, b, c, b1, MD4S12); /* 2 */
  MD4FF (c, d, a, b, b2, MD4S13); /* 3 */
  MD4FF (b, c, d, a, b3, MD4S14); /* 4 */
  MD4FF (a, b, c, d, b4, MD4S11); /* 5 */
  MD4FF (d, a, b, c, b5, MD4S12); /* 6 */
  MD4FF (c, d, a, b, b6, MD4S13); /* 7 */
  MD4FF (b, c, d, a, b7, MD4S14); /* 8 */
  MD4FF (a, b, c, d, b8, MD4S11); /* 9 */
  MD4FF (d, a, b, c, b9, MD4S12); /* 10 */
  MD4FF (c, d, a, b, b10, MD4S13); /* 11 */
  MD4FF (b, c, d, a, b11, MD4S14); /* 12 */
  MD4FF (a, b, c, d, b12, MD4S11); /* 13 */
  MD4FF (d, a, b, c, b13, MD4S12); /* 14 */
  MD4FF (c, d, a, b, b14, MD4S13); /* 15 */
  MD4FF (b, c, d, a, b15, MD4S14); /* 16 */

  /* Round 2 */
  MD4GG (a, b, c, d, b0, MD4S21); /* 17 */
  MD4GG (d, a, b, c, b4, MD4S22); /* 18 */
  MD4GG (c, d, a, b, b8, MD4S23); /* 19 */
  MD4GG (b, c, d, a, b12, MD4S24); /* 20 */
  MD4GG (a, b, c, d, b1, MD4S21); /* 21 */
  MD4GG (d, a, b, c, b5, MD4S22); /* 22 */
  MD4GG (c, d, a, b, b9, MD4S23); /* 23 */
  MD4GG (b, c, d, a, b13, MD4S24); /* 24 */
  MD4GG (a, b, c, d, b2, MD4S21); /* 25 */
  MD4GG (d, a, b, c, b6, MD4S22); /* 26 */
  MD4GG (c, d, a, b, b10, MD4S23); /* 27 */
  MD4GG (b, c, d, a, b14, MD4S24); /* 28 */
  MD4GG (a, b, c, d, b3, MD4S21); /* 29 */
  MD4GG (d, a, b, c, b7, MD4S22); /* 30 */
  MD4GG (c, d, a, b, b11, MD4S23); /* 31 */
  MD4GG (b, c, d, a, b15, MD4S24); /* 32 */


  /* Round 3 */
  MD4HH (a, b, c, d, b0, MD4S31); /* 33 */
  MD4HH (d, a, b, c, b8, MD4S32); /* 34 */
  MD4HH (c, d, a, b, b4, MD4S33); /* 35 */
  MD4HH (b, c, d, a, b12, MD4S34); /* 36 */
  MD4HH (a, b, c, d, b2, MD4S31); /* 37 */
  MD4HH (d, a, b, c, b10, MD4S32); /* 38 */
  MD4HH (c, d, a, b, b6, MD4S33); /* 39 */
  MD4HH (b, c, d, a, b14, MD4S34); /* 40 */
  MD4HH (a, b, c, d, b1, MD4S31); /* 41 */
  MD4HH (d, a, b, c, b9, MD4S32); /* 42 */
  MD4HH (c, d, a, b, b5, MD4S33); /* 43 */
  MD4HH (b, c, d, a, b13, MD4S34); /* 44 */
  MD4HH (a, b, c, d, b3, MD4S31); /* 45 */
  MD4HH (d, a, b, c, b11, MD4S32); /* 46 */
  MD4HH (c, d, a, b, b7, MD4S33); /* 47 */
  MD4HH (b, c, d, a, b15, MD4S34); /* 48 */

  // Finally, add initial values, as this is the only pass we make.
  a += 0x67452301;
  b += 0xefcdab89;
  c += 0x98badcfe;
  d += 0x10325476;
}