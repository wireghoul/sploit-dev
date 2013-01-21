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
 * @section DESCRIPTION
 *
 * This file includes common SHA256 defines/functions/useful code.
 * 
 * This is used for forward and reverse SHA256 as needed at various points in
 * code.  It should be usable on GPUs as well, though may require additional
 * mutations/defines for OpenCL to use bitselect/etc.
 */


#ifndef __CH_SHA256_H__
#define __CH_SHA256_H__

#define SHR(x,n) ((x & 0xFFFFFFFF) >> n)
#define ROTR(x,n) (SHR(x,n) | (x << (32 - n)))

#define S0(x) (ROTR(x, 7) ^ ROTR(x,18) ^  SHR(x, 3))
#define S1(x) (ROTR(x,17) ^ ROTR(x,19) ^  SHR(x,10))

#define S2(x) (ROTR(x, 2) ^ ROTR(x,13) ^ ROTR(x,22))
#define S3(x) (ROTR(x, 6) ^ ROTR(x,11) ^ ROTR(x,25))

#define F0(x,y,z) ((x & y) | (z & (x | y)))
#define F1(x,y,z) (z ^ (x & (y ^ z)))

#define R(t)                                    \
(                                               \
    W[t] = S1(W[t -  2]) + W[t -  7] +          \
           S0(W[t - 15]) + W[t - 16]            \
)

#define P(a,b,c,d,e,f,g,h,x,K)                  \
{                                               \
    temp1 = h + S3(e) + F1(e,f,g) + K + x;      \
    temp2 = S2(a) + F0(a,b,c);                  \
    d += temp1; h = temp1 + temp2;              \
}

#define reverse(x)(x>>24)|((x<<8) & 0x00FF0000)|((x>>8) & 0x0000FF00)|(x<<24);



#define SHA256_FULL() { \
    uint32_t temp1, temp2; \
    a = 0x6A09E667; \
    b = 0xBB67AE85; \
    c = 0x3C6EF372; \
    d = 0xA54FF53A; \
    e = 0x510E527F; \
    f = 0x9B05688C; \
    g = 0x1F83D9AB; \
    h = 0x5BE0CD19; \
    P( a, b, c, d, e, f, g, h,  b0, 0x428A2F98 ); \
    P( h, a, b, c, d, e, f, g,  b1, 0x71374491 ); \
    P( g, h, a, b, c, d, e, f,  b2, 0xB5C0FBCF ); \
    P( f, g, h, a, b, c, d, e,  b3, 0xE9B5DBA5 ); \
    P( e, f, g, h, a, b, c, d,  b4, 0x3956C25B ); \
    P( d, e, f, g, h, a, b, c,  b5, 0x59F111F1 ); \
    P( c, d, e, f, g, h, a, b,  b6, 0x923F82A4 ); \
    P( b, c, d, e, f, g, h, a,  b7, 0xAB1C5ED5 ); \
    P( a, b, c, d, e, f, g, h,  b8, 0xD807AA98 ); \
    P( h, a, b, c, d, e, f, g,  b9, 0x12835B01 ); \
    P( g, h, a, b, c, d, e, f, b10, 0x243185BE ); \
    P( f, g, h, a, b, c, d, e, b11, 0x550C7DC3 ); \
    P( e, f, g, h, a, b, c, d, b12, 0x72BE5D74 ); \
    P( d, e, f, g, h, a, b, c, b13, 0x80DEB1FE ); \
    P( c, d, e, f, g, h, a, b, b14, 0x9BDC06A7 ); \
    P( b, c, d, e, f, g, h, a, b15, 0xC19BF174 ); \
    b0 = S1(b14) + b9 + S0(b1) + b0; \
    P( a, b, c, d, e, f, g, h,  b0, 0xE49B69C1 ); \
    b1 = S1(b15) + b10 + S0(b2) + b1; \
    P( h, a, b, c, d, e, f, g,  b1, 0xEFBE4786 ); \
    b2 = S1(b0) + b11 + S0(b3) + b2; \
    P( g, h, a, b, c, d, e, f,  b2, 0x0FC19DC6 ); \
    b3 = S1(b1) + b12 + S0(b4) + b3; \
    P( f, g, h, a, b, c, d, e,  b3, 0x240CA1CC ); \
    b4 = S1(b2) + b13 + S0(b5) + b4; \
    P( e, f, g, h, a, b, c, d,  b4, 0x2DE92C6F ); \
    b5 = S1(b3) + b14 + S0(b6) + b5; \
    P( d, e, f, g, h, a, b, c,  b5, 0x4A7484AA ); \
    b6 = S1(b4) + b15 + S0(b7) + b6; \
    P( c, d, e, f, g, h, a, b,  b6, 0x5CB0A9DC ); \
    b7 = S1(b5) + b0 + S0(b8) + b7; \
    P( b, c, d, e, f, g, h, a,  b7, 0x76F988DA ); \
    b8 = S1(b6) + b1 + S0(b9) + b8; \
    P( a, b, c, d, e, f, g, h,  b8, 0x983E5152 ); \
    b9 = S1(b7) + b2 + S0(b10) + b9; \
    P( h, a, b, c, d, e, f, g,  b9, 0xA831C66D ); \
    b10 = S1(b8) + b3 + S0(b11) + b10; \
    P( g, h, a, b, c, d, e, f, b10, 0xB00327C8 ); \
    b11 = S1(b9) + b4 + S0(b12) + b11; \
    P( f, g, h, a, b, c, d, e, b11, 0xBF597FC7 ); \
    b12 = S1(b10) + b5 + S0(b13) + b12; \
    P( e, f, g, h, a, b, c, d, b12, 0xC6E00BF3 ); \
    b13 = S1(b11) + b6 + S0(b14) + b13; \
    P( d, e, f, g, h, a, b, c, b13, 0xD5A79147 ); \
    b14 = S1(b12) + b7 + S0(b15) + b14; \
    P( c, d, e, f, g, h, a, b, b14, 0x06CA6351 ); \
    b15 = S1(b13) + b8 + S0(b0) + b15; \
    P( b, c, d, e, f, g, h, a, b15, 0x14292967 ); \
    b0 = S1(b14) + b9 + S0(b1) + b0; \
    P( a, b, c, d, e, f, g, h,  b0, 0x27B70A85 ); \
    b1 = S1(b15) + b10 + S0(b2) + b1; \
    P( h, a, b, c, d, e, f, g,  b1, 0x2E1B2138 ); \
    b2 = S1(b0) + b11 + S0(b3) + b2; \
    P( g, h, a, b, c, d, e, f,  b2, 0x4D2C6DFC ); \
    b3 = S1(b1) + b12 + S0(b4) + b3; \
    P( f, g, h, a, b, c, d, e,  b3, 0x53380D13 ); \
    b4 = S1(b2) + b13 + S0(b5) + b4; \
    P( e, f, g, h, a, b, c, d,  b4, 0x650A7354 ); \
    b5 = S1(b3) + b14 + S0(b6) + b5; \
    P( d, e, f, g, h, a, b, c,  b5, 0x766A0ABB ); \
    b6 = S1(b4) + b15 + S0(b7) + b6; \
    P( c, d, e, f, g, h, a, b,  b6, 0x81C2C92E ); \
    b7 = S1(b5) + b0 + S0(b8) + b7; \
    P( b, c, d, e, f, g, h, a,  b7, 0x92722C85 ); \
    b8 = S1(b6) + b1 + S0(b9) + b8; \
    P( a, b, c, d, e, f, g, h,  b8, 0xA2BFE8A1 ); \
    b9 = S1(b7) + b2 + S0(b10) + b9; \
    P( h, a, b, c, d, e, f, g,  b9, 0xA81A664B ); \
    b10 = S1(b8) + b3 + S0(b11) + b10; \
    P( g, h, a, b, c, d, e, f, b10, 0xC24B8B70 ); \
    b11 = S1(b9) + b4 + S0(b12) + b11; \
    P( f, g, h, a, b, c, d, e, b11, 0xC76C51A3 ); \
    b12 = S1(b10) + b5 + S0(b13) + b12; \
    P( e, f, g, h, a, b, c, d, b12, 0xD192E819 ); \
    b13 = S1(b11) + b6 + S0(b14) + b13; \
    P( d, e, f, g, h, a, b, c, b13, 0xD6990624 ); \
    b14 = S1(b12) + b7 + S0(b15) + b14; \
    P( c, d, e, f, g, h, a, b, b14, 0xF40E3585 ); \
    b15 = S1(b13) + b8 + S0(b0) + b15; \
    P( b, c, d, e, f, g, h, a, b15, 0x106AA070 ); \
    b0 = S1(b14) + b9 + S0(b1) + b0; \
    P( a, b, c, d, e, f, g, h,  b0, 0x19A4C116 ); \
    b1 = S1(b15) + b10 + S0(b2) + b1; \
    P( h, a, b, c, d, e, f, g,  b1, 0x1E376C08 ); \
    b2 = S1(b0) + b11 + S0(b3) + b2; \
    P( g, h, a, b, c, d, e, f,  b2, 0x2748774C ); \
    b3 = S1(b1) + b12 + S0(b4) + b3; \
    P( f, g, h, a, b, c, d, e,  b3, 0x34B0BCB5 ); \
    b4 = S1(b2) + b13 + S0(b5) + b4; \
    P( e, f, g, h, a, b, c, d,  b4, 0x391C0CB3 ); \
    b5 = S1(b3) + b14 + S0(b6) + b5; \
    P( d, e, f, g, h, a, b, c,  b5, 0x4ED8AA4A ); \
    b6 = S1(b4) + b15 + S0(b7) + b6; \
    P( c, d, e, f, g, h, a, b,  b6, 0x5B9CCA4F ); \
    b7 = S1(b5) + b0 + S0(b8) + b7; \
    P( b, c, d, e, f, g, h, a,  b7, 0x682E6FF3 ); \
    b8 = S1(b6) + b1 + S0(b9) + b8; \
    P( a, b, c, d, e, f, g, h,  b8, 0x748F82EE ); \
    b9 = S1(b7) + b2 + S0(b10) + b9; \
    P( h, a, b, c, d, e, f, g,  b9, 0x78A5636F ); \
    b10 = S1(b8) + b3 + S0(b11) + b10; \
    P( g, h, a, b, c, d, e, f, b10, 0x84C87814 ); \
    b11 = S1(b9) + b4 + S0(b12) + b11; \
    P( f, g, h, a, b, c, d, e, b11, 0x8CC70208 ); \
    b12 = S1(b10) + b5 + S0(b13) + b12; \
    P( e, f, g, h, a, b, c, d, b12, 0x90BEFFFA ); \
    b13 = S1(b11) + b6 + S0(b14) + b13; \
    P( d, e, f, g, h, a, b, c, b13, 0xA4506CEB ); \
    b14 = S1(b12) + b7 + S0(b15) + b14; \
    P( c, d, e, f, g, h, a, b, b14, 0xBEF9A3F7 ); \
    b15 = S1(b13) + b8 + S0(b0) + b15; \
    P( b, c, d, e, f, g, h, a, b15, 0xC67178F2 ); \
    a += 0x6A09E667; \
    b += 0xBB67AE85; \
    c += 0x3C6EF372; \
    d += 0xA54FF53A; \
    e += 0x510E527F; \
    f += 0x9B05688C; \
    g += 0x1F83D9AB; \
    h += 0x5BE0CD19; \
}

#endif