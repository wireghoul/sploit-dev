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

#include "Multiforcer_Common/CHCommon.h"

extern struct global_commands global_interface;


typedef uint32_t UINT4;
__device__ __constant__ char deviceCharset[MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN];
__device__ __constant__ __align__(16) unsigned char charsetLengths[16];
__device__ __constant__ unsigned char constantBitmap[8192]; // for lookups
__device__ __constant__ uint32_t MSSQLSalts[MAX_MSSQL_HASHES];

#include "Multiforcer_CUDA_device/CUDAcommon.h"



#define ROL32(_val32, _nBits) (((_val32)<<(_nBits))|((_val32)>>(32-(_nBits))))

#define SHABLK(a,b,c,d) (a = ROL32(b ^ c ^ d ^ a,1))
#define SHABLK0(i) (i = (ROL32(i,24) & 0xFF00FF00) | (ROL32(i,8) & 0x00FF00FF))

#define _R0(v,w,x,y,z,i) {z+=((w&(x^y))^y)+i+0x5A827999+ROL32(v,5); w=ROL32(w,30); }
#define _R0_NULL(v,w,x,y,z) {z+=((w&(x^y))^y)+0x5A827999+ROL32(v,5); w=ROL32(w,30); }
#define _R1(v,w,x,y,z,i) { z+=((w&(x^y))^y)+i+0x5A827999+ROL32(v,5);w=ROL32(w,30); }
#define _R2(v,w,x,y,z,i) { z+=(w^x^y)+i+0x6ED9EBA1+ROL32(v,5);w=ROL32(w,30); }
#define _R3(v,w,x,y,z,i) { z+=(((w|x)&y)|(w&x))+i+0x8F1BBCDC+ROL32(v,5); w=ROL32(w,30); }
#define _R4(v,w,x,y,z,i) { z+=(w^x^y)+i+0xCA62C1D6+ROL32(v,5); w=ROL32(w,30); }


#define reverse(x)(x>>24)|((x<<8) & 0x00FF0000)|((x>>8) & 0x0000FF00)|(x<<24);


///assumes the hash length is smaller than 11 characters. BIG UGLY MACRO, DONT FUCK WITH IT. ALSO HIGLY OPTIMIZED.
#define SHA_TRANSFORM_SMALL(a,b,c,d,e,DATA0,DATA1,DATA2,DATA3,DATA4,DATA5,DATA6,DATA7,DATA8,DATA9,DATA10,DATA11,DATA12,DATA13,DATA14,DATA15)\
		;a = 0x67452301; b = 0xEFCDAB89; c = 0x98BADCFE; d = 0x10325476; e = 0xC3D2E1F0; \
        SHABLK0(DATA0);SHABLK0(DATA1);SHABLK0(DATA2);SHABLK0(DATA3);SHABLK0(DATA15);\
		_R0(a,b,c,d,e, DATA0); _R0(e,a,b,c,d, DATA1);\
		_R0(d,e,a,b,c, DATA2); _R0(c,d,e,a,b, DATA3); \
		_R0_NULL(b,c,d,e,a); _R0_NULL(a,b,c,d,e);\
		_R0_NULL(e,a,b,c,d); _R0_NULL(d,e,a,b,c); \
		_R0_NULL(c,d,e,a,b); _R0_NULL(b,c,d,e,a);\
		_R0_NULL(a,b,c,d,e); _R0_NULL(e,a,b,c,d); \
		_R0_NULL(d,e,a,b,c); _R0_NULL(c,d,e,a,b);\
		_R0_NULL(b,c,d,e,a); _R0(a,b,c,d,e,DATA15); \
		DATA0 = ROL32(DATA0 ^ DATA2,1);	DATA1 = ROL32(DATA1 ^ DATA3,1);\
		DATA2 = ROL32(DATA2 ^ DATA15,1);DATA3 = ROL32(DATA0 ,1);\
		DATA4 = ROL32(DATA1 ,1);DATA5 = ROL32(DATA2 ,1);\
		DATA6 = ROL32(DATA3 ,1);DATA7 = ROL32(DATA15 ^ DATA4 ,1);\
		DATA8 = ROL32(DATA0 ^ DATA5 ,1);DATA9 = ROL32(DATA1 ^ DATA6 ,1);\
		DATA10 = ROL32(DATA2 ^ DATA7 ,1);DATA11 = ROL32(DATA3 ^ DATA8 ,1);\
		DATA12 = ROL32(DATA4 ^ DATA9 ,1);DATA13 = ROL32(DATA5 ^ DATA10 ^ DATA15 ,1);\
		DATA14 = ROL32(DATA0 ^ DATA6 ^ DATA11 ,1);DATA15 = ROL32(DATA1 ^ DATA7 ^ DATA12 ^ DATA15,1);\
		SHA_TRANSFORM_END(a,b,c,d,e,DATA0,DATA1,DATA2,DATA3,DATA4,DATA5,DATA6,DATA7,DATA8,DATA9,DATA10,DATA11,DATA12,DATA13,DATA14,DATA15);

///assumes the hash length is smaller than 56 characters. BIG UGLY MACRO, DONT FUCK WITH IT. ALSO HIGLY OPTIMIZED.
#define SHA_TRANSFORM(a,b,c,d,e,DATA0,DATA1,DATA2,DATA3,DATA4,DATA5,DATA6,DATA7,DATA8,DATA9,DATA10,DATA11,DATA12,DATA13,DATA14,DATA15)\
		;a = 0x67452301; b = 0xEFCDAB89; c = 0x98BADCFE; d = 0x10325476; e = 0xC3D2E1F0; \
        SHABLK0(DATA0);SHABLK0(DATA1);SHABLK0(DATA2);SHABLK0(DATA3);\
		SHABLK0(DATA4);SHABLK0(DATA5);SHABLK0(DATA6);SHABLK0(DATA7);\
		SHABLK0(DATA8);SHABLK0(DATA9);SHABLK0(DATA10);SHABLK0(DATA11);\
		SHABLK0(DATA12);SHABLK0(DATA13);SHABLK0(DATA14);SHABLK0(DATA15);\
		_R0(a,b,c,d,e, DATA0); _R0(e,a,b,c,d, DATA1);\
		_R0(d,e,a,b,c, DATA2); _R0(c,d,e,a,b, DATA3); \
		_R0(b,c,d,e,a, DATA4); _R0(a,b,c,d,e, DATA5);\
		_R0(e,a,b,c,d, DATA6); _R0(d,e,a,b,c, DATA7); \
		_R0(c,d,e,a,b, DATA8); _R0(b,c,d,e,a, DATA9);\
		_R0(a,b,c,d,e,DATA10); _R0(e,a,b,c,d,DATA11); \
		_R0(d,e,a,b,c,DATA12); _R0(c,d,e,a,b,DATA13);\
		_R0(b,c,d,e,a,DATA14); _R0(a,b,c,d,e,DATA15); \
		SHABLK(DATA0,DATA13,DATA8,DATA2); SHABLK(DATA1,DATA14,DATA9,DATA3);\
		SHABLK(DATA2,DATA15,DATA10,DATA4);SHABLK(DATA3,DATA0,DATA11,DATA5);  \
		SHABLK(DATA4,DATA1,DATA12,DATA6); SHABLK(DATA5,DATA2,DATA13,DATA7);\
		SHABLK(DATA6,DATA3,DATA14,DATA8); SHABLK(DATA7,DATA4,DATA15,DATA9); \
		SHABLK(DATA8,DATA5,DATA0,DATA10); SHABLK(DATA9,DATA6,DATA1,DATA11);\
		SHABLK(DATA10,DATA7,DATA2,DATA12);SHABLK(DATA11,DATA8,DATA3,DATA13);  \
		SHABLK(DATA12,DATA9,DATA4,DATA14);SHABLK(DATA13,DATA10,DATA5,DATA15);\
		SHABLK(DATA14,DATA11,DATA6,DATA0);SHABLK(DATA15,DATA12,DATA7,DATA1); \
		SHA_TRANSFORM_END(a,b,c,d,e,DATA0,DATA1,DATA2,DATA3,DATA4,DATA5,DATA6,DATA7,DATA8,DATA9,DATA10,DATA11,DATA12,DATA13,DATA14,DATA15);


///as the starting bit of routines for different sized hashes changes, this is the common part.
#define SHA_TRANSFORM_END(a,b,c,d,e,DATA0,DATA1,DATA2,DATA3,DATA4,DATA5,DATA6,DATA7,DATA8,DATA9,DATA10,DATA11,DATA12,DATA13,DATA14,DATA15)\
		;_R1(e,a,b,c,d,DATA0);_R1(d,e,a,b,c,DATA1);_R1(c,d,e,a,b,DATA2);_R1(b,c,d,e,a,DATA3);\
		_R2(a,b,c,d,e,DATA4); _R2(e,a,b,c,d,DATA5); _R2(d,e,a,b,c,DATA6);_R2(c,d,e,a,b,DATA7);\
		_R2(b,c,d,e,a,DATA8);_R2(a,b,c,d,e,DATA9);_R2(e,a,b,c,d,DATA10);_R2(d,e,a,b,c,DATA11);\
		_R2(c,d,e,a,b,DATA12);_R2(b,c,d,e,a,DATA13);_R2(a,b,c,d,e,DATA14);_R2(e,a,b,c,d,DATA15);\
		SHABLK(DATA0,DATA13,DATA8,DATA2);_R2(d,e,a,b,c,DATA0);SHABLK(DATA1,DATA14,DATA9,DATA3);_R2(c,d,e,a,b,DATA1); \
		SHABLK(DATA2,DATA15,DATA10,DATA4);_R2(b,c,d,e,a,DATA2);SHABLK(DATA3,DATA0,DATA11,DATA5); _R2(a,b,c,d,e,DATA3); \
		SHABLK(DATA4,DATA1,DATA12,DATA6);_R2(e,a,b,c,d,DATA4);SHABLK(DATA5,DATA2,DATA13,DATA7); _R2(d,e,a,b,c,DATA5); \
		SHABLK(DATA6,DATA3,DATA14,DATA8);_R2(c,d,e,a,b,DATA6);SHABLK(DATA7,DATA4,DATA15,DATA9); _R2(b,c,d,e,a,DATA7); \
		SHABLK(DATA8,DATA5,DATA0,DATA10);_R3(a,b,c,d,e,DATA8);SHABLK(DATA9,DATA6,DATA1,DATA11); _R3(e,a,b,c,d,DATA9);\
		SHABLK(DATA10,DATA7,DATA2,DATA12);_R3(d,e,a,b,c,DATA10);SHABLK(DATA11,DATA8,DATA3,DATA13); _R3(c,d,e,a,b,DATA11); \
		SHABLK(DATA12,DATA9,DATA4,DATA14);_R3(b,c,d,e,a,DATA12);SHABLK(DATA13,DATA10,DATA5,DATA15); _R3(a,b,c,d,e,DATA13);\
		SHABLK(DATA14,DATA11,DATA6,DATA0);_R3(e,a,b,c,d,DATA14);SHABLK(DATA15,DATA12,DATA7,DATA1); _R3(d,e,a,b,c,DATA15); \
		SHABLK(DATA0,DATA13,DATA8,DATA2); _R3(c,d,e,a,b,DATA0);SHABLK(DATA1,DATA14,DATA9,DATA3); _R3(b,c,d,e,a,DATA1);\
		SHABLK(DATA2,DATA15,DATA10,DATA4);_R3(a,b,c,d,e,DATA2);SHABLK(DATA3,DATA0,DATA11,DATA5); _R3(e,a,b,c,d,DATA3); \
		SHABLK(DATA4,DATA1,DATA12,DATA6);_R3(d,e,a,b,c,DATA4); SHABLK(DATA5,DATA2,DATA13,DATA7);_R3(c,d,e,a,b,DATA5); \
		SHABLK(DATA6,DATA3,DATA14,DATA8);_R3(b,c,d,e,a,DATA6);SHABLK(DATA7,DATA4,DATA15,DATA9); _R3(a,b,c,d,e,DATA7); \
		SHABLK(DATA8,DATA5,DATA0,DATA10);_R3(e,a,b,c,d,DATA8);SHABLK(DATA9,DATA6,DATA1,DATA11); _R3(d,e,a,b,c,DATA9); \
		SHABLK(DATA10,DATA7,DATA2,DATA12);_R3(c,d,e,a,b,DATA10);SHABLK(DATA11,DATA8,DATA3,DATA13); _R3(b,c,d,e,a,DATA11);\
		SHABLK(DATA12,DATA9,DATA4,DATA14);_R4(a,b,c,d,e,DATA12);SHABLK(DATA13,DATA10,DATA5,DATA15); _R4(e,a,b,c,d,DATA13);\
		SHABLK(DATA14,DATA11,DATA6,DATA0);_R4(d,e,a,b,c,DATA14);SHABLK(DATA15,DATA12,DATA7,DATA1); _R4(c,d,e,a,b,DATA15); \
		SHABLK(DATA0,DATA13,DATA8,DATA2);_R4(b,c,d,e,a,DATA0);SHABLK(DATA1,DATA14,DATA9,DATA3); _R4(a,b,c,d,e,DATA1); \
		SHABLK(DATA2,DATA15,DATA10,DATA4);_R4(e,a,b,c,d,DATA2);SHABLK(DATA3,DATA0,DATA11,DATA5); _R4(d,e,a,b,c,DATA3); \
        SHABLK(DATA4,DATA1,DATA12,DATA6);_R4(c,d,e,a,b,DATA4); SHABLK(DATA5,DATA2,DATA13,DATA7);_R4(b,c,d,e,a,DATA5); \
		SHABLK(DATA6,DATA3,DATA14,DATA8);_R4(a,b,c,d,e,DATA6);SHABLK(DATA7,DATA4,DATA15,DATA9); _R4(e,a,b,c,d,DATA7); \
		SHABLK(DATA8,DATA5,DATA0,DATA10);_R4(d,e,a,b,c,DATA8);SHABLK(DATA9,DATA6,DATA1,DATA11); _R4(c,d,e,a,b,DATA9);\
		SHABLK(DATA10,DATA7,DATA2,DATA12);_R4(b,c,d,e,a,DATA10);SHABLK(DATA11,DATA8,DATA3,DATA13); _R4(a,b,c,d,e,DATA11); \
		SHABLK(DATA12,DATA9,DATA4,DATA14);_R4(e,a,b,c,d,DATA12); SHABLK(DATA13,DATA10,DATA5,DATA15);_R4(d,e,a,b,c,DATA13); \
		SHABLK(DATA14,DATA11,DATA6,DATA0);_R4(c,d,e,a,b,DATA14);SHABLK(DATA15,DATA12,DATA7,DATA1); _R4(b,c,d,e,a,DATA15); \
		a=0x67452301 + a;b=0xEFCDAB89 + b;c= 0x98BADCFE + c;d=0x10325476 + d;e=0xC3D2E1F0 + e; \
		DATA0=reverse(a);DATA1=reverse(b);DATA2=reverse(c);DATA3=reverse(d);DATA4=reverse(e);



#define CREATE_BUFFER(BUFFER_NAME)DWORD BUFFER_NAME##0  (0),BUFFER_NAME##1   (0),BUFFER_NAME##2	 (0),BUFFER_NAME##3 (0),\
													       BUFFER_NAME##4  (0),BUFFER_NAME##5   (0),BUFFER_NAME##6    (0),BUFFER_NAME##7 (0),\
													       BUFFER_NAME##8  (0),BUFFER_NAME##9   (0),BUFFER_NAME##10   (0),BUFFER_NAME##11 (0),\
														   BUFFER_NAME##12 (0),BUFFER_NAME##13  (0),BUFFER_NAME##14;

#define INIT_BUFFER(BUFFER_NAME,DATA)                     DWORD * m_buffer=(DWORD*)DATA;\
													      BUFFER_NAME##0=m_buffer[0];   BUFFER_NAME##1=m_buffer[1];  BUFFER_NAME##2=m_buffer[2];\
														  BUFFER_NAME##3=m_buffer[3];   BUFFER_NAME##4=m_buffer[4];  BUFFER_NAME##5=m_buffer[5];\
														  BUFFER_NAME##6=m_buffer[6];   BUFFER_NAME##7=m_buffer[7];  BUFFER_NAME##8=m_buffer[8];\
														  BUFFER_NAME##9=m_buffer[9];   BUFFER_NAME##10=m_buffer[10];BUFFER_NAME##11=m_buffer[11];\
														  BUFFER_NAME##12=m_buffer[12]; BUFFER_NAME##13=m_buffer[13];BUFFER_NAME##14=m_buffer[14];




//look, we can convert a dword to a string fastah! no if statements involved.
//Yes i got bored.
#define FAST_HEX_TO_STRING(STRING,n,INDEX,UCASE)\
										STRING[1+INDEX]=48+(n&15)      + (((n&15)>9)*(7+(32*UCASE)));\
										STRING[0+INDEX]=48+((n>>4)&15) + ((((n>>4)&15)>9)*(7+(32*UCASE)));\
										STRING[3+INDEX]=48+((n>>8)&15) + ((((n>>8)&15)>9)*(7+(32*UCASE)));\
										STRING[2+INDEX]=48+((n>>12)&15)+ ((((n>>12)&15)>9)*(7+(32*UCASE)));\
										STRING[5+INDEX]=48+((n>>16)&15)+ ((((n>>16)&15)>9)*(7+(32*UCASE)));\
										STRING[4+INDEX]=48+((n>>20)&15)+ ((((n>>20)&15)>9)*(7+(32*UCASE)));\
										STRING[7+INDEX]=48+((n>>24)&15)+ ((((n>>24)&15)>9)*(7+(32*UCASE)));\
										STRING[6+INDEX]=48+((n>>28)&15)+ ((((n>>28)&15)>9)*(7+(32*UCASE)));


// Init MSSQL functions.
__device__ inline void initMSSQL(int length, unsigned char *sharedCharset,
		unsigned char &p0, unsigned char &p1, unsigned char &p2, unsigned char &p3,
		unsigned char &p4, unsigned char &p5, unsigned char &p6, unsigned char &p7,
		unsigned char &p8, unsigned char &p9, unsigned char &p10, unsigned char &p11,
		unsigned char &p12, unsigned char &p13, unsigned char &p14, unsigned char &p15,
		UINT4 &b0, UINT4 &b1, UINT4 &b2, UINT4 &b3, UINT4 &b4, UINT4 &b5, UINT4 &b6, UINT4 &b7,
		UINT4 &b8, UINT4 &b9, UINT4 &b10, UINT4 &b11, UINT4 &b12, UINT4 &b13, UINT4 &b14, UINT4 &b15,
                uint32_t *sharedMSSQLSalt, uint32_t hash_to_check) {

  clearB0toB15(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);

  // Set length properly (length in bits)
  // Note that this is /different/ from MD!
  // Add 4 bytes for the salt!
  b15 = ((length * 2) + 4) * 8 << 24;

 if (length == 0) {
    b1 |= 0x00000080;
    return;
  }
  b0 |= sharedCharset[p0];
  if (length == 1) {
    // Set the salt byte
    b0 &= 0x0000ffff;
    b0 |= ((sharedMSSQLSalt[hash_to_check]) & 0xffff) << 16;
    b1 &= 0xffff0000;
    b1 |= ((sharedMSSQLSalt[hash_to_check] & 0xffff0000) >> 16);

    b1 |= 0x00800000;
    return;
  }
  b0 |= sharedCharset[p1 + MAX_CHARSET_LENGTH] << 16;
  if (length == 2) {
    b1 = sharedMSSQLSalt[hash_to_check];
    b2 |= 0x00000080;
	return;
  }
  b1 |= sharedCharset[p2 + MAX_CHARSET_LENGTH * 2];
  if (length == 3) {
    // Set the salt byte
    // Mask off the high bits - we have the low bits already.
    b1 &= 0x0000ffff;
    b1 |= ((sharedMSSQLSalt[hash_to_check]) & 0xffff) << 16;
    // Mask off the high bits of b2
    b2 &= 0xffff0000;
    b2 |= ((sharedMSSQLSalt[hash_to_check] & 0xffff0000) >> 16);
    b2 |= 0x00800000;
    return;
  }
  b1 |= sharedCharset[p3 + MAX_CHARSET_LENGTH * 3] << 16;
  if (length == 4) {
    b2 = sharedMSSQLSalt[hash_to_check];
    b3 |= 0x00000080;
    return;
  }
  b2 |= sharedCharset[p4 + MAX_CHARSET_LENGTH * 4];
  if (length == 5) {
    b2 &= 0x0000ffff;
    b2 |= ((sharedMSSQLSalt[hash_to_check]) & 0xffff) << 16;
    b3 &= 0xffff0000;
    b3 |= ((sharedMSSQLSalt[hash_to_check] & 0xffff0000) >> 16);
    b3 |= 0x00800000;
    return;
  }
  b2 |= sharedCharset[p5 + (MAX_CHARSET_LENGTH * 5)] << 16;
  if (length == 6) {
    b3 = sharedMSSQLSalt[hash_to_check];
    b4 |= 0x00000080;
    return;
  }
  b3 |= sharedCharset[p6 + (MAX_CHARSET_LENGTH * 6)];
  if (length == 7) {
    b3 &= 0x0000ffff;
    b3 |= ((sharedMSSQLSalt[hash_to_check]) & 0xffff) << 16;
    b4 &= 0xffff0000;
    b4 |= ((sharedMSSQLSalt[hash_to_check] & 0xffff0000) >> 16);
    b4 |= 0x00800000;
    return;
  }
  b3 |= sharedCharset[p7 + (MAX_CHARSET_LENGTH * 7)] << 16;
  if (length == 8) {
    b4 = sharedMSSQLSalt[hash_to_check];
    b5 |= 0x00000080;
    return;
  }
  b4 |= sharedCharset[p8 + (MAX_CHARSET_LENGTH * 8)];
  if (length == 9) {
    b4 &= 0x0000ffff;
    b4 |= ((sharedMSSQLSalt[hash_to_check]) & 0xffff) << 16;
    b5 &= 0xffff0000;
    b5 |= ((sharedMSSQLSalt[hash_to_check] & 0xffff0000) >> 16);
    b5 |= 0x00800000;
    return;
  }
  b4 |= sharedCharset[p9 + (MAX_CHARSET_LENGTH * 9)] << 16;
  if (length == 10) {
    b5 = sharedMSSQLSalt[hash_to_check];
    b6 |= 0x00000080;
    return;
  }
  b5 |= sharedCharset[p10 + (MAX_CHARSET_LENGTH * 10)];
  if (length == 11) {
    b5 &= 0x0000ffff;
    b5 |= ((sharedMSSQLSalt[hash_to_check]) & 0xffff) << 16;
    b6 &= 0xffff0000;
    b7 |= ((sharedMSSQLSalt[hash_to_check] & 0xffff0000) >> 16);
    b6 |= 0x00800000;
    return;
  }
  b5 |= sharedCharset[p11 + (MAX_CHARSET_LENGTH * 11)] << 16;
  if (length == 12) {
    b6 = sharedMSSQLSalt[hash_to_check];
    b7 |= 0x00000080;
    return;
  }
  b6 |= sharedCharset[p12 + (MAX_CHARSET_LENGTH * 12)];
  if (length == 13) {
    b6 &= 0x0000ffff;
    b7 |= ((sharedMSSQLSalt[hash_to_check]) & 0xffff) << 16;
    b7 &= 0xffff0000;
    b7 |= ((sharedMSSQLSalt[hash_to_check] & 0xffff0000) >> 16);
    b7 |= 0x00800000;
    return;
  }
  b6 |= sharedCharset[p13 + (MAX_CHARSET_LENGTH * 13)] << 16;
  if (length == 14) {
    b7 = sharedMSSQLSalt[hash_to_check];
    b8 |= 0x00000080;
    return;
  }
  b7 |= sharedCharset[p14 + (MAX_CHARSET_LENGTH * 14)];
  if (length == 15) {
    b7 &= 0x0000ffff;
    b7 |= ((sharedMSSQLSalt[hash_to_check]) & 0xffff) << 16;
    b8 &= 0xffff0000;
    b8 |= ((sharedMSSQLSalt[hash_to_check] & 0xffff0000) >> 16);
    b8 |= 0x00800000;
    return;
  }
  b7 |= sharedCharset[p15 + (MAX_CHARSET_LENGTH * 15)] << 16;
  if (length == 16) {
    b8 = sharedMSSQLSalt[hash_to_check];
    b9 |= 0x00000080;
    return;
  }
}






__device__ inline void copyMSSQLSalts(uint32_t *sharedMSSQLSalts) {
  int a;

  for (a = 0; a <= ((MAX_MSSQL_HASHES) / blockDim.x); a++) {
    if (((a * blockDim.x) + threadIdx.x) < (1024)) {
        sharedMSSQLSalts[(a * blockDim.x) + threadIdx.x] = MSSQLSalts[(a * blockDim.x) + threadIdx.x];
    }
  }
}


__device__ inline void checkHashMultiMSSQL(int pass_length, unsigned char *sharedBitmap, unsigned char *DEVICE_HashTable, uint32_t numberOfPasswords,
		uint32_t *DEVICE_Hashes_32, unsigned char *success, unsigned char *OutputPassword,
		unsigned char p0, unsigned char p1, unsigned char p2, unsigned char p3,
		unsigned char p4, unsigned char p5, unsigned char p6, unsigned char p7,
		unsigned char p8, unsigned char p9, unsigned char p10, unsigned char p11,
		unsigned char p12, unsigned char p13, unsigned char p14, unsigned char p15,
		UINT4 a, UINT4 b, UINT4 c, UINT4 d, UINT4 e,
		uint32_t &search_index, uint32_t &search_high, uint32_t &search_low, uint32_t &hash_order_a,
		uint32_t &hash_order_mem, uint32_t &temp, uint32_t hash_being_checked) {
  //uint32_t search_index, search_high, search_low, hash_order_a, hash_order_mem, temp;
  if ((sharedBitmap[(a & 0x0000ffff) >> 3] >> (a & 0x00000007)) & 0x00000001) {
      if ((!DEVICE_HashTable) || (DEVICE_HashTable && (DEVICE_HashTable[a >> 3] >> (a & 0x00000007)) & 0x00000001)) {
  // Init binary search through global password space

  search_high = numberOfPasswords;
  search_low = 0;
  search_index = 0;
  while (search_low < search_high) {
    // Midpoint between search_high and search_low
    search_index = search_low + (search_high - search_low) / 2;
    // reorder from low endian to big endian for searching, as hashes are sorted by byte.
    temp = DEVICE_Hashes_32[5 * search_index];
    hash_order_mem = (temp & 0xff) << 24 | ((temp >> 8) & 0xff) << 16 | ((temp >> 16) & 0xff) << 8 | ((temp >> 24) & 0xff);
    hash_order_a = (a & 0xff) << 24 | ((a >> 8) & 0xff) << 16 | ((a >> 16) & 0xff) << 8 | ((a >> 24) & 0xff);

    // Adjust search_high & search_low to work through space
    if (hash_order_mem < hash_order_a) {
      search_low = search_index + 1;
    } else  {
      search_high = search_index;
    }
    if ((hash_order_a == hash_order_mem) && (search_low < numberOfPasswords)) {
      // Break out of the search loop - search_index is on a value
      break;
    }
  }

  // Yes - it's a goto.  And it speeds up performance significantly (15%).
  // It stays.  These values are already loaded.  If they are not the same,
  // there is NO point to touching global memory again.
  if (hash_order_a != hash_order_mem) {
    goto next;
  }
  // We've broken out of the loop, search_index should be on a matching value
  // Loop while the search index is the same - linear search through this to find all possible
  // matching passwords.
  // We first need to move backwards to the beginning, as we may be in the middle of a set of matching hashes.
  // If we are index 0, do NOT subtract, as we will wrap and this goes poorly.

  while (search_index && (a == DEVICE_Hashes_32[(search_index - 1) * 5])) {
    search_index--;
  }


  while ((a == DEVICE_Hashes_32[search_index * 5])) {
  if (a == DEVICE_Hashes_32[search_index * 5]) {
    if (b == DEVICE_Hashes_32[search_index * 5 + 1]) {
      if (c == DEVICE_Hashes_32[search_index * 5 + 2]) {
        if (d == DEVICE_Hashes_32[search_index * 5 + 3]) {
        if (e == DEVICE_Hashes_32[search_index * 5 + 4]) {
        if (pass_length >= 1) OutputPassword[search_index * MAX_PASSWORD_LEN + 0] = deviceCharset[p0];
        if (pass_length >= 2) OutputPassword[search_index * MAX_PASSWORD_LEN + 1] = deviceCharset[p1 + (MAX_CHARSET_LENGTH * 1)];
        if (pass_length >= 3) OutputPassword[search_index * MAX_PASSWORD_LEN + 2] = deviceCharset[p2 + (MAX_CHARSET_LENGTH * 2)];
        if (pass_length >= 4) OutputPassword[search_index * MAX_PASSWORD_LEN + 3] = deviceCharset[p3 + (MAX_CHARSET_LENGTH * 3)];
        if (pass_length >= 5) OutputPassword[search_index * MAX_PASSWORD_LEN + 4] = deviceCharset[p4 + (MAX_CHARSET_LENGTH * 4)];
        if (pass_length >= 6) OutputPassword[search_index * MAX_PASSWORD_LEN + 5] = deviceCharset[p5 + (MAX_CHARSET_LENGTH * 5)];
        if (pass_length >= 7) OutputPassword[search_index * MAX_PASSWORD_LEN + 6] = deviceCharset[p6 + (MAX_CHARSET_LENGTH * 6)];
        if (pass_length >= 8) OutputPassword[search_index * MAX_PASSWORD_LEN + 7] = deviceCharset[p7 + (MAX_CHARSET_LENGTH * 7)];
        if (pass_length >= 9) OutputPassword[search_index * MAX_PASSWORD_LEN + 8] = deviceCharset[p8 + (MAX_CHARSET_LENGTH * 8)];
        if (pass_length >= 10) OutputPassword[search_index * MAX_PASSWORD_LEN + 9] = deviceCharset[p9 + (MAX_CHARSET_LENGTH * 9)];
        if (pass_length >= 11) OutputPassword[search_index * MAX_PASSWORD_LEN + 10] = deviceCharset[p10 + (MAX_CHARSET_LENGTH * 10)];
        if (pass_length >= 12) OutputPassword[search_index * MAX_PASSWORD_LEN + 11] = deviceCharset[p11 + (MAX_CHARSET_LENGTH * 11)];
        if (pass_length >= 13) OutputPassword[search_index * MAX_PASSWORD_LEN + 12] = deviceCharset[p12 + (MAX_CHARSET_LENGTH * 12)];
        if (pass_length >= 14) OutputPassword[search_index * MAX_PASSWORD_LEN + 13] = deviceCharset[p13 + (MAX_CHARSET_LENGTH * 13)];
        if (pass_length >= 15) OutputPassword[search_index * MAX_PASSWORD_LEN + 14] = deviceCharset[p14 + (MAX_CHARSET_LENGTH * 14)];
        if (pass_length >= 16) OutputPassword[search_index * MAX_PASSWORD_LEN + 15] = deviceCharset[p15 + (MAX_CHARSET_LENGTH * 15)];
        // Final check to make sure the seed matches
        if (hash_being_checked == search_index) {
            success[search_index] = (unsigned char) 1;
        }
        }
        }
      }
    }
  }
  search_index++;
  }
  }
  }
  // This is where the goto goes.  Notice the skipping of all the global memory access.
  next:
  return;
}



#define CUDA_MSSQL_KERNEL_CREATELONG(length) \
__global__ void CUDA_MSSQL_Search_##length (unsigned char *OutputPassword, unsigned char *success, \
			    int charsetLen, uint32_t numberOfPasswords, struct start_positions *DEVICE_Start_Positions, unsigned int count, unsigned char * DEVICE_Hashes, unsigned char *DEVICE_HashTable) { \
  const int pass_length = length; \
  uint32_t b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15; \
  uint32_t a,b,c,d,e; \
  uint32_t thread_index = blockIdx.x*blockDim.x + threadIdx.x; \
  uint32_t *DEVICE_Hashes_32 = (uint32_t *)DEVICE_Hashes; \
  unsigned char p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15; \
  UINT4 password_count = 0; \
  uint32_t hash_to_check = 0; \
  __shared__ __align__(16) unsigned char sharedCharset[MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN_16]; \
  __shared__ __align__(16) unsigned char sharedBitmap[8192]; \
  __shared__ __align__(8) unsigned char sharedLengths[16];  \
  __shared__ uint32_t sharedMSSQLSalts[MAX_MSSQL_HASHES]; \
  copyMSSQLSalts(sharedMSSQLSalts); \
  copyCharsetAndBitmap(sharedCharset, sharedBitmap, sharedLengths, charsetLen, pass_length); \
  loadStartPositions(pass_length, thread_index, DEVICE_Start_Positions,  \
		   p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15); \
  while (password_count < count) { \
    for (hash_to_check = 0; hash_to_check < numberOfPasswords; hash_to_check++) { \
      initMSSQL(pass_length, sharedCharset, \
  	p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, \
	b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, \
        sharedMSSQLSalts, hash_to_check); \
      SHA_TRANSFORM(a, b, c, d, e, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      checkHashMultiMSSQL(pass_length, sharedBitmap, DEVICE_HashTable, numberOfPasswords, \
		DEVICE_Hashes_32, success, OutputPassword, \
		p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, \
		b0, b1, b2, b3, b4, a, b, c, d, e, b14, hash_to_check); \
     } \
  password_count++; \
  incrementCounters##length##Multi(); \
  } \
}

CUDA_MSSQL_KERNEL_CREATELONG(1)
CUDA_MSSQL_KERNEL_CREATELONG(2)
CUDA_MSSQL_KERNEL_CREATELONG(3)
CUDA_MSSQL_KERNEL_CREATELONG(4)
CUDA_MSSQL_KERNEL_CREATELONG(5)
CUDA_MSSQL_KERNEL_CREATELONG(6)
CUDA_MSSQL_KERNEL_CREATELONG(7)
CUDA_MSSQL_KERNEL_CREATELONG(8)
CUDA_MSSQL_KERNEL_CREATELONG(9)
CUDA_MSSQL_KERNEL_CREATELONG(10)
CUDA_MSSQL_KERNEL_CREATELONG(11)
CUDA_MSSQL_KERNEL_CREATELONG(12)
CUDA_MSSQL_KERNEL_CREATELONG(13)
CUDA_MSSQL_KERNEL_CREATELONG(14)
CUDA_MSSQL_KERNEL_CREATELONG(15)
CUDA_MSSQL_KERNEL_CREATELONG(16)


/*
__global__ void CUDA_MSSQL_Search_3 (unsigned char *OutputPassword, unsigned char *success,
			    int charsetLen, uint32_t numberOfPasswords, struct start_positions *DEVICE_Start_Positions, unsigned int count, unsigned char * DEVICE_Hashes, unsigned char *DEVICE_HashTable) { 
  const int pass_length = 3;
  uint32_t b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15; 
  uint32_t a,b,c,d, e;
  uint32_t thread_index = blockIdx.x*blockDim.x + threadIdx.x;
  uint32_t *DEVICE_Hashes_32 = (uint32_t *)DEVICE_Hashes; 
  unsigned char p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15; 
  UINT4 password_count = 0;
  uint32_t hash_to_check = 0;
  __shared__ __align__(16) unsigned char sharedCharset[MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN];
  __shared__ __align__(16) unsigned char sharedBitmap[8192]; 
  __shared__ __align__(8) unsigned char sharedLengths[16];
  __shared__ uint32_t sharedMSSQLSalts[1024];
  copyMSSQLSalts(sharedMSSQLSalts);
  copyCharsetAndBitmap(sharedCharset, sharedBitmap, sharedLengths, charsetLen, pass_length);
  
  loadStartPositions(pass_length, thread_index, DEVICE_Start_Positions,  
		   p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15); 
  while (password_count < count) { 
  for (hash_to_check = 0; hash_to_check < numberOfPasswords; hash_to_check++) {
  initMSSQL(pass_length, sharedCharset,
  	p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
	b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15,
        sharedMSSQLSalts, hash_to_check);
      SHA_TRANSFORM(a, b, c, d, e, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
      checkHashMultiMSSQL(pass_length, sharedBitmap, DEVICE_HashTable, numberOfPasswords,
		DEVICE_Hashes_32, success, OutputPassword, 
		p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, 
		b0, b1, b2, b3, b4, a, b, c, d, e, b5); 
    }
  password_count++;
  incrementCounters3Multi(); 
  } 
}


__global__ void CUDA_MSSQL_Search_4 (unsigned char *OutputPassword, unsigned char *success,
			    int charsetLen, uint32_t numberOfPasswords, struct start_positions *DEVICE_Start_Positions, unsigned int count, unsigned char * DEVICE_Hashes, unsigned char *DEVICE_HashTable) {
  const int pass_length = 4;
  uint32_t b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;
  uint32_t a,b,c,d, e;
  uint32_t thread_index = blockIdx.x*blockDim.x + threadIdx.x;
  uint32_t *DEVICE_Hashes_32 = (uint32_t *)DEVICE_Hashes;
  unsigned char p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15;
  UINT4 password_count = 0;
  uint32_t hash_to_check = 0;
  __shared__ __align__(16) unsigned char sharedCharset[MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN];
  __shared__ __align__(16) unsigned char sharedBitmap[8192];
  __shared__ __align__(8) unsigned char sharedLengths[16];
  __shared__ uint32_t sharedMSSQLSalts[1024];
  copyMSSQLSalts(sharedMSSQLSalts);
  copyCharsetAndBitmap(sharedCharset, sharedBitmap, sharedLengths, charsetLen, pass_length);

  loadStartPositions(pass_length, thread_index, DEVICE_Start_Positions,
		   p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15);
  while (password_count < count) {
  for (hash_to_check = 0; hash_to_check < numberOfPasswords; hash_to_check++) {
  initMSSQL(pass_length, sharedCharset,
  	p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
	b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15,
        sharedMSSQLSalts, hash_to_check);
      SHA_TRANSFORM(a, b, c, d, e, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
      checkHashMultiMSSQL(pass_length, sharedBitmap, DEVICE_HashTable, numberOfPasswords,
		DEVICE_Hashes_32, success, OutputPassword,
		p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
		b0, b1, b2, b3, b4, a, b, c, d, e, b5);
    }
  password_count++;
  incrementCounters4Multi();
  }
}
*/


// Copy the shared variables to the host
extern "C" void copyMSSQLDataToConstant(char *hostCharset, int charsetLength,
        unsigned char *hostCharsetLengths, unsigned char *hostSharedBitmap,
        int threadId, uint32_t *salts) {
    //printf("Thread %d in CHHashTypeNTLM.cu, copyNTLMDataToCharset()\n", threadId);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceCharset, hostCharset, (MAX_CHARSET_LENGTH * charsetLength)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(constantBitmap, hostSharedBitmap, 8192));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(charsetLengths, hostCharsetLengths, 16));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MSSQLSalts, salts, MAX_MSSQL_HASHES * sizeof(uint32_t)));
}

extern "C" void Launch_CUDA_MSSQL_Kernel(int passlength, uint64_t charsetLength, int numberOfPasswords, unsigned char *DEVICE_Passwords,
        unsigned char *DEVICE_Success, struct start_positions *DEVICE_Start_Positions, uint64_t per_step, uint64_t threads, uint64_t blocks, unsigned char *DEVICE_Hashes, unsigned char *DEVICE_Bitmap) {
        if (passlength == 1) {
	  CUDA_MSSQL_Search_1 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 2) {
	  CUDA_MSSQL_Search_2 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 3) {
	  CUDA_MSSQL_Search_3 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 4) {
	  CUDA_MSSQL_Search_4 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 5) {
	  CUDA_MSSQL_Search_5 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 6) {
	  CUDA_MSSQL_Search_6 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 7) {
	  CUDA_MSSQL_Search_7 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 8) {
	  CUDA_MSSQL_Search_8 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 9) {
	  CUDA_MSSQL_Search_9 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 10) {
	  CUDA_MSSQL_Search_10 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 11) {
	  CUDA_MSSQL_Search_11 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 12) {
	  CUDA_MSSQL_Search_12 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 13) {
	  CUDA_MSSQL_Search_13 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 14) {
	  CUDA_MSSQL_Search_14 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 15) {
	  CUDA_MSSQL_Search_15 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	} else if (passlength == 16) {
	  CUDA_MSSQL_Search_16 <<< blocks, threads >>> (DEVICE_Passwords, DEVICE_Success, charsetLength, numberOfPasswords, DEVICE_Start_Positions, per_step, DEVICE_Hashes, DEVICE_Bitmap);
	}

	cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
      {
        sprintf(global_interface.exit_message, "Cuda error: %s.\n", cudaGetErrorString( err) );
        global_interface.exit = 1;
        return;
      }
}