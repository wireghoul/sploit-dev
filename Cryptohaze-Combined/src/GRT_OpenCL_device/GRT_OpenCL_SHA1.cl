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

// Using these SHA1 routines.  Vampyr, if you have an issue with this, let me know.
// Based on the "free as in freedom; do not use in a commercial product" comment.


//this file is part of (P)assrape by vampyr_6@hotmail.com
///set of macros for an extemely optimized SHA routine.
//
///coding powered by Grendel-Harsh Generation and Celldweller - Switchback (Neuroticfish Razorblade Remix)
///and lots of caffeine of course.
///averages about 79.5 million hashes/s on a gforce 8800 320mb.
///A note to the warry coder: integer divisions and multiplications are expensive.
///However, if you have 24-bits integers, you could use __mul24 and __umul24 for 4-cycle multiplication and division.
///Use bitshifts wherever possible. If n is a power of 2, (i/n) is equivalent to (i>>log2(n)) and (i%n) is equivalent to (i&(n-1)).
///Do NOT use functions that take char's or short's as arguments, they need to be converted to integers before use.
///
///Good memory acces patterns are also of utmost importance. An example:
///1: Load data from device memory to shared memory.
///2: Synchronize with all the other threads of the block.
///3: Process the data in shared memory.
///4: Synchronize again if necessary to make sure that shared memory has been updated with the results.
///5: Write the results back to device memory.
///
///Align structures with __align__(16)

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


void padSHAHash(int length,
                vector_type *b0, vector_type *b1, vector_type *b2, vector_type *b3, vector_type *b4, vector_type *b5, vector_type *b6, vector_type *b7,
		vector_type *b8, vector_type *b9, vector_type *b10, vector_type *b11, vector_type *b12, vector_type *b13, vector_type *b14, vector_type *b15);



inline void padSHAHash(int length,
                vector_type *b0, vector_type *b1, vector_type *b2, vector_type *b3, vector_type *b4, vector_type *b5, vector_type *b6, vector_type *b7,
		vector_type *b8, vector_type *b9, vector_type *b10, vector_type *b11, vector_type *b12, vector_type *b13, vector_type *b14, vector_type *b15) {

  // Set length properly (length in bits)
  *b15 = (vector_type)(((length * 8) & 0xff) << 24 | (((length * 8) >> 8) & 0xff) << 16);

  if (length == 0) {
    *b0 |= (vector_type)0x00000080;
    return;
  }
  if (length == 1) {
    *b0 |= (vector_type)0x00008000;
    return;
  }
  if (length == 2) {
    *b0 |= (vector_type)0x00800000;
    return;
  }
  if (length == 3) {
    *b0 |= (vector_type)0x80000000;
    return;
  }
  if (length == 4) {
    *b1 |= (vector_type)0x00000080;
    return;
  }
  if (length == 5) {
    *b1 |= (vector_type)0x00008000;
    return;
  }
  if (length == 6) {
    *b1 |= (vector_type)0x00800000;
    return;
  }
  if (length == 7) {
    *b1 |= (vector_type)0x80000000;
    return;
  }
  if (length == 8) {
    *b2 |= (vector_type)0x00000080;
    return;
  }
  if (length == 9) {
    *b2 |= (vector_type)0x00008000;
    return;
  }
  if (length == 10) {
    *b2 |= (vector_type)0x00800000;
    return;
  }
  if (length == 11) {
   *b2 |= (vector_type)0x80000000;
    return;
  }
  if (length == 12) {
    *b3 |= (vector_type)0x00000080;
    return;
  }
  if (length == 13) {
    *b3 |= (vector_type)0x00008000;
    return;
  }
  if (length == 14) {
    *b3 |= (vector_type)0x00800000;
    return;
  }
  if (length == 15) {
    *b3 |= (vector_type)0x80000000;
    return;
  }
  if (length == 16) {
    *b4 |= (vector_type)0x00000080;
    return;
  }
  if (length == 17) {
    *b4 |= (vector_type)0x00008000;
    return;
  }
  if (length == 18) {
    *b4 |= (vector_type)0x00800000;
    return;
  }
  if (length == 19) {
    *b4 |= (vector_type)0x80000000;
    return;
  }
  if (length == 20) {
    *b5 |= (vector_type)0x00000080;
    return;
  }
  if (length == 21) {
    *b5 |= (vector_type)0x00008000;
    return;
  }
  if (length == 22) {
    *b5 |= (vector_type)0x00800000;
    return;
  }
  if (length == 23) {
    *b5 |= (vector_type)0x80000000;
    return;
  }
  if (length == 24) {
    *b6 |= (vector_type)0x00000080;
    return;
  }
  if (length == 25) {
    *b6 |= (vector_type)0x00008000;
    return;
  }
  if (length == 26) {
    *b6 |= (vector_type)0x00800000;
    return;
  }
  if (length == 27) {
    *b6 |= (vector_type)0x80000000;
    return;
  }
  if (length == 28) {
    *b7 |= (vector_type)0x00000080;
    return;
  }
  if (length == 29) {
    *b7 |= (vector_type)0x00008000;
    return;
  }
  if (length == 30) {
    *b7 |= (vector_type)0x00800000;
    return;
  }
  if (length == 31) {
    *b7 |= (vector_type)0x80000000;
    return;
  }

}

