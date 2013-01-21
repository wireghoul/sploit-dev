
#if BITALIGN
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#define MD4ROTATE_LEFT(x, y) amd_bitalign(x, x, (uint)(32 - y))
#define MD4FF(a,b,c,d,x,s) { (a) = (a)+x+(amd_bytealign((b),(c),(d))); (a) = MD4ROTATE_LEFT((a), (s)); }
#define MD4H(x, y, z) ((x) ^ (y) ^ (z))
#define MD4GG(a,b,c,d,x,s) {(a) = (a) + (vector_type)0x5a827999 + (amd_bytealign((b), ((d) | (c)), ((c) & (d)))) +x  ; (a) = MD4ROTATE_LEFT((a), (s)); }
#elif NVIDIA_HACKS
#define MD4ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))
#define MD4F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define MD4G(x, y, z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
#define MD4H(x, y, z) ((x) ^ (y) ^ (z))
#define MD4FF(a, b, c, d, x, s) { \
    (a) += MD4F ((b), (c), (d)) + (x); \
    (a) = MD4ROTATE_LEFT ((a), (s)); \
  }
#define MD4GG(a, b, c, d, x, s) { \
    (a) += MD4G ((b), (c), (d)) + (x) + (vector_type)0x5a827999; \
    (a) = MD4ROTATE_LEFT ((a), (s)); \
  }
#else
#define MD4ROTATE_LEFT(x, y) rotate((vector_type)x, (uint)y)
#define MD4F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define MD4G(x, y, z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
#define MD4H(x, y, z) ((x) ^ (y) ^ (z))
#define MD4FF(a, b, c, d, x, s) { \
    (a) += MD4F ((b), (c), (d)) + (x); \
    (a) = MD4ROTATE_LEFT ((a), (s)); \
  }
#define MD4GG(a, b, c, d, x, s) { \
    (a) += MD4G ((b), (c), (d)) + (x) + (vector_type)0x5a827999; \
    (a) = MD4ROTATE_LEFT ((a), (s)); \
  }
#endif


// Hash defines
/* MD4 Defines as per RFC reference implementation */
/*#define MD4F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define MD4G(x, y, z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
#define MD4H(x, y, z) ((x) ^ (y) ^ (z))
#define MD4ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))
*/
/*
#define MD4FF(a, b, c, d, x, s) { \
    (a) += MD4F ((b), (c), (d)) + (x); \
    (a) = MD4ROTATE_LEFT ((a), (s)); \
  }
*/
/*
#define MD4GG(a, b, c, d, x, s) { \
    (a) += MD4G ((b), (c), (d)) + (x) + (vector_type)0x5a827999; \
    (a) = MD4ROTATE_LEFT ((a), (s)); \
  }
*/
#define MD4HH(a, b, c, d, x, s) { \
    (a) += MD4H ((b), (c), (d)) + (x) + (vector_type)0x6ed9eba1; \
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





// Make the Apple compiler happy...
void OpenCL_MD4(vector_type b0, vector_type b1, vector_type b2, vector_type b3, vector_type b4, vector_type b5, vector_type b6, vector_type b7,
			   vector_type b8, vector_type b9, vector_type b10, vector_type b11, vector_type b12, vector_type b13, vector_type b14, vector_type b15,
			   vector_type *a, vector_type *b, vector_type *c, vector_type *d);


void reduceSingleCharsetNTLM(vector_type *b0, vector_type *b1, vector_type *b2,
        vector_type *b3, vector_type *b4, vector_type *b5,
        vector_type a, vector_type b, vector_type c, vector_type d,
        uint CurrentStep, __local unsigned char *charset, uint charset_offset, int PasswordLength, uint Device_Table_Index);







inline void OpenCL_MD4(vector_type b0, vector_type b1, vector_type b2, vector_type b3, vector_type b4, vector_type b5, vector_type b6, vector_type b7,
			   vector_type b8, vector_type b9, vector_type b10, vector_type b11, vector_type b12, vector_type b13, vector_type b14, vector_type b15,
			   vector_type *a, vector_type *b, vector_type *c, vector_type *d) {
  *a = (vector_type)0x67452301;
  *b = (vector_type)0xefcdab89;
  *c = (vector_type)0x98badcfe;
  *d = (vector_type)0x10325476;

  MD4FF (*a, *b, *c, *d, b0, MD4S11); /* 1 */
  MD4FF (*d, *a, *b, *c, b1, MD4S12); /* 2 */
  MD4FF (*c, *d, *a, *b, b2, MD4S13); /* 3 */
  MD4FF (*b, *c, *d, *a, b3, MD4S14); /* 4 */
  MD4FF (*a, *b, *c, *d, b4, MD4S11); /* 5 */
  MD4FF (*d, *a, *b, *c, b5, MD4S12); /* 6 */
  MD4FF (*c, *d, *a, *b, b6, MD4S13); /* 7 */
  MD4FF (*b, *c, *d, *a, b7, MD4S14); /* 8 */
  MD4FF (*a, *b, *c, *d, b8, MD4S11); /* 9 */
  MD4FF (*d, *a, *b, *c, b9, MD4S12); /* 10 */
  MD4FF (*c, *d, *a, *b, b10, MD4S13); /* 11 */
  MD4FF (*b, *c, *d, *a, b11, MD4S14); /* 12 */
  MD4FF (*a, *b, *c, *d, b12, MD4S11); /* 13 */
  MD4FF (*d, *a, *b, *c, b13, MD4S12); /* 14 */
  MD4FF (*c, *d, *a, *b, b14, MD4S13); /* 15 */
  MD4FF (*b, *c, *d, *a, b15, MD4S14); /* 16 */

  /* Round 2 */
  MD4GG (*a, *b, *c, *d, b0, MD4S21); /* 17 */
  MD4GG (*d, *a, *b, *c, b4, MD4S22); /* 18 */
  MD4GG (*c, *d, *a, *b, b8, MD4S23); /* 19 */
  MD4GG (*b, *c, *d, *a, b12, MD4S24); /* 20 */
  MD4GG (*a, *b, *c, *d, b1, MD4S21); /* 21 */
  MD4GG (*d, *a, *b, *c, b5, MD4S22); /* 22 */
  MD4GG (*c, *d, *a, *b, b9, MD4S23); /* 23 */
  MD4GG (*b, *c, *d, *a, b13, MD4S24); /* 24 */
  MD4GG (*a, *b, *c, *d, b2, MD4S21); /* 25 */
  MD4GG (*d, *a, *b, *c, b6, MD4S22); /* 26 */
  MD4GG (*c, *d, *a, *b, b10, MD4S23); /* 27 */
  MD4GG (*b, *c, *d, *a, b14, MD4S24); /* 28 */
  MD4GG (*a, *b, *c, *d, b3, MD4S21); /* 29 */
  MD4GG (*d, *a, *b, *c, b7, MD4S22); /* 30 */
  MD4GG (*c, *d, *a, *b, b11, MD4S23); /* 31 */
  MD4GG (*b, *c, *d, *a, b15, MD4S24); /* 32 */


  /* Round 3 */
  MD4HH (*a, *b, *c, *d, b0, MD4S31); /* 33 */
  MD4HH (*d, *a, *b, *c, b8, MD4S32); /* 34 */
  MD4HH (*c, *d, *a, *b, b4, MD4S33); /* 35 */
  MD4HH (*b, *c, *d, *a, b12, MD4S34); /* 36 */
  MD4HH (*a, *b, *c, *d, b2, MD4S31); /* 37 */
  MD4HH (*d, *a, *b, *c, b10, MD4S32); /* 38 */
  MD4HH (*c, *d, *a, *b, b6, MD4S33); /* 39 */
  MD4HH (*b, *c, *d, *a, b14, MD4S34); /* 40 */
  MD4HH (*a, *b, *c, *d, b1, MD4S31); /* 41 */
  MD4HH (*d, *a, *b, *c, b9, MD4S32); /* 42 */
  MD4HH (*c, *d, *a, *b, b5, MD4S33); /* 43 */
  MD4HH (*b, *c, *d, *a, b13, MD4S34); /* 44 */
  MD4HH (*a, *b, *c, *d, b3, MD4S31); /* 45 */
  MD4HH (*d, *a, *b, *c, b11, MD4S32); /* 46 */
  MD4HH (*c, *d, *a, *b, b7, MD4S33); /* 47 */
  MD4HH (*b, *c, *d, *a, b15, MD4S34); /* 48 */

  // Finally, add initial values, as this is the only pass we make.
  *a += (vector_type)0x67452301;
  *b += (vector_type)0xefcdab89;
  *c += (vector_type)0x98badcfe;
  *d += (vector_type)0x10325476;
}




inline void reduceSingleCharsetNTLM(vector_type *b0, vector_type *b1, vector_type *b2,
        vector_type *b3, vector_type *b4, vector_type *b5,
        vector_type a, vector_type b, vector_type c, vector_type d,
        uint CurrentStep, __local unsigned char *charset, uint charset_offset, int PasswordLength, uint Device_Table_Index) {

    vector_type z;
    // Reduce it
    // First 3
    z = (a+(vector_type)CurrentStep+(vector_type)Device_Table_Index) % (vector_type)(256*256*256);
#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b0).s0 = (uint)charset[(z.s0 % 256) + charset_offset];
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b0).s1 = (uint)charset[(z.s1 % 256) + charset_offset];
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b0).s2 = (uint)charset[(z.s2 % 256) + charset_offset];
    (*b0).s3 = (uint)charset[(z.s3 % 256) + charset_offset];
#endif
#if grt_vector_8 || grt_vector_16
    (*b0).s4 = (uint)charset[(z.s4 % 256) + charset_offset];
    (*b0).s5 = (uint)charset[(z.s5 % 256) + charset_offset];
    (*b0).s6 = (uint)charset[(z.s6 % 256) + charset_offset];
    (*b0).s7 = (uint)charset[(z.s7 % 256) + charset_offset];
#endif
#if grt_vector_16
    (*b0).s8 = (uint)charset[(z.s8 % 256) + charset_offset];
    (*b0).s9 = (uint)charset[(z.s9 % 256) + charset_offset];
    (*b0).sA = (uint)charset[(z.sA % 256) + charset_offset];
    (*b0).sB = (uint)charset[(z.sB % 256) + charset_offset];
    (*b0).sC = (uint)charset[(z.sC % 256) + charset_offset];
    (*b0).sD = (uint)charset[(z.sD % 256) + charset_offset];
    (*b0).sE = (uint)charset[(z.sE % 256) + charset_offset];
    (*b0).sF = (uint)charset[(z.sF % 256) + charset_offset];
#endif
    if (PasswordLength == 1) {return;}

    z /= (vector_type)256;
#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b0).s0 |= (uint)charset[(z.s0 % 256) + charset_offset] << 16;
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b0).s1 |= (uint)charset[(z.s1 % 256) + charset_offset] << 16;
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b0).s2 |= (uint)charset[(z.s2 % 256) + charset_offset] << 16;
    (*b0).s3 |= (uint)charset[(z.s3 % 256) + charset_offset] << 16;
#endif
#if grt_vector_8 || grt_vector_16
    (*b0).s4 |= (uint)charset[(z.s4 % 256) + charset_offset] << 16;
    (*b0).s5 |= (uint)charset[(z.s5 % 256) + charset_offset] << 16;
    (*b0).s6 |= (uint)charset[(z.s6 % 256) + charset_offset] << 16;
    (*b0).s7 |= (uint)charset[(z.s7 % 256) + charset_offset] << 16;
#endif
#if grt_vector_16
    (*b0).s8 |= (uint)charset[(z.s8 % 256) + charset_offset] << 16;
    (*b0).s9 |= (uint)charset[(z.s9 % 256) + charset_offset] << 16;
    (*b0).sA |= (uint)charset[(z.sA % 256) + charset_offset] << 16;
    (*b0).sB |= (uint)charset[(z.sB % 256) + charset_offset] << 16;
    (*b0).sC |= (uint)charset[(z.sC % 256) + charset_offset] << 16;
    (*b0).sD |= (uint)charset[(z.sD % 256) + charset_offset] << 16;
    (*b0).sE |= (uint)charset[(z.sE % 256) + charset_offset] << 16;
    (*b0).sF |= (uint)charset[(z.sF % 256) + charset_offset] << 16;
#endif
    if (PasswordLength == 2) {return;}

    z /= (vector_type)256;
#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b1).s0 = (uint)charset[(z.s0 % 256) + charset_offset];
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b1).s1 = (uint)charset[(z.s1 % 256) + charset_offset];
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b1).s2 = (uint)charset[(z.s2 % 256) + charset_offset];
    (*b1).s3 = (uint)charset[(z.s3 % 256) + charset_offset];
#endif
#if grt_vector_8 || grt_vector_16
    (*b1).s4 = (uint)charset[(z.s4 % 256) + charset_offset];
    (*b1).s5 = (uint)charset[(z.s5 % 256) + charset_offset];
    (*b1).s6 = (uint)charset[(z.s6 % 256) + charset_offset];
    (*b1).s7 = (uint)charset[(z.s7 % 256) + charset_offset];
#endif
#if grt_vector_16
    (*b1).s8 = (uint)charset[(z.s8 % 256) + charset_offset];
    (*b1).s9 = (uint)charset[(z.s9 % 256) + charset_offset];
    (*b1).sA = (uint)charset[(z.sA % 256) + charset_offset];
    (*b1).sB = (uint)charset[(z.sB % 256) + charset_offset];
    (*b1).sC = (uint)charset[(z.sC % 256) + charset_offset];
    (*b1).sD = (uint)charset[(z.sD % 256) + charset_offset];
    (*b1).sE = (uint)charset[(z.sE % 256) + charset_offset];
    (*b1).sF = (uint)charset[(z.sF % 256) + charset_offset];
#endif
    if (PasswordLength == 3) {return;}

    // Second 3
    z = (b+(vector_type)CurrentStep+(vector_type)Device_Table_Index) % (vector_type)(256*256*256);
#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b1).s0 |= (uint)charset[(z.s0 % 256) + charset_offset] << 16;
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b1).s1 |= (uint)charset[(z.s1 % 256) + charset_offset] << 16;
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b1).s2 |= (uint)charset[(z.s2 % 256) + charset_offset] << 16;
    (*b1).s3 |= (uint)charset[(z.s3 % 256) + charset_offset] << 16;
#endif
#if grt_vector_8 || grt_vector_16
    (*b1).s4 |= (uint)charset[(z.s4 % 256) + charset_offset] << 16;
    (*b1).s5 |= (uint)charset[(z.s5 % 256) + charset_offset] << 16;
    (*b1).s6 |= (uint)charset[(z.s6 % 256) + charset_offset] << 16;
    (*b1).s7 |= (uint)charset[(z.s7 % 256) + charset_offset] << 16;
#endif
#if grt_vector_16
    (*b1).s8 |= (uint)charset[(z.s8 % 256) + charset_offset] << 16;
    (*b1).s9 |= (uint)charset[(z.s9 % 256) + charset_offset] << 16;
    (*b1).sA |= (uint)charset[(z.sA % 256) + charset_offset] << 16;
    (*b1).sB |= (uint)charset[(z.sB % 256) + charset_offset] << 16;
    (*b1).sC |= (uint)charset[(z.sC % 256) + charset_offset] << 16;
    (*b1).sD |= (uint)charset[(z.sD % 256) + charset_offset] << 16;
    (*b1).sE |= (uint)charset[(z.sE % 256) + charset_offset] << 16;
    (*b1).sF |= (uint)charset[(z.sF % 256) + charset_offset] << 16;
#endif
    if (PasswordLength == 4) {return;}

    z /= (vector_type)256;

#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b2).s0 = (uint)charset[(z.s0 % 256) + charset_offset];
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b2).s1 = (uint)charset[(z.s1 % 256) + charset_offset];
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b2).s2 = (uint)charset[(z.s2 % 256) + charset_offset];
    (*b2).s3 = (uint)charset[(z.s3 % 256) + charset_offset];
#endif
#if grt_vector_8 || grt_vector_16
    (*b2).s4 = (uint)charset[(z.s4 % 256) + charset_offset];
    (*b2).s5 = (uint)charset[(z.s5 % 256) + charset_offset];
    (*b2).s6 = (uint)charset[(z.s6 % 256) + charset_offset];
    (*b2).s7 = (uint)charset[(z.s7 % 256) + charset_offset];
#endif
#if grt_vector_16
    (*b2).s8 = (uint)charset[(z.s8 % 256) + charset_offset];
    (*b2).s9 = (uint)charset[(z.s9 % 256) + charset_offset];
    (*b2).sA = (uint)charset[(z.sA % 256) + charset_offset];
    (*b2).sB = (uint)charset[(z.sB % 256) + charset_offset];
    (*b2).sC = (uint)charset[(z.sC % 256) + charset_offset];
    (*b2).sD = (uint)charset[(z.sD % 256) + charset_offset];
    (*b2).sE = (uint)charset[(z.sE % 256) + charset_offset];
    (*b2).sF = (uint)charset[(z.sF % 256) + charset_offset];
#endif
    if (PasswordLength == 5) {return;}

    z /= (vector_type)256;
#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b2).s0 |= (uint)charset[(z.s0 % 256) + charset_offset] << 16;
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b2).s1 |= (uint)charset[(z.s1 % 256) + charset_offset] << 16;
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b2).s2 |= (uint)charset[(z.s2 % 256) + charset_offset] << 16;
    (*b2).s3 |= (uint)charset[(z.s3 % 256) + charset_offset] << 16;
#endif
#if grt_vector_8 || grt_vector_16
    (*b2).s4 |= (uint)charset[(z.s4 % 256) + charset_offset] << 16;
    (*b2).s5 |= (uint)charset[(z.s5 % 256) + charset_offset] << 16;
    (*b2).s6 |= (uint)charset[(z.s6 % 256) + charset_offset] << 16;
    (*b2).s7 |= (uint)charset[(z.s7 % 256) + charset_offset] << 16;
#endif
#if grt_vector_16
    (*b2).s8 |= (uint)charset[(z.s8 % 256) + charset_offset] << 16;
    (*b2).s9 |= (uint)charset[(z.s9 % 256) + charset_offset] << 16;
    (*b2).sA |= (uint)charset[(z.sA % 256) + charset_offset] << 16;
    (*b2).sB |= (uint)charset[(z.sB % 256) + charset_offset] << 16;
    (*b2).sC |= (uint)charset[(z.sC % 256) + charset_offset] << 16;
    (*b2).sD |= (uint)charset[(z.sD % 256) + charset_offset] << 16;
    (*b2).sE |= (uint)charset[(z.sE % 256) + charset_offset] << 16;
    (*b2).sF |= (uint)charset[(z.sF % 256) + charset_offset] << 16;
#endif
    if (PasswordLength == 6) {return;}

    // Last 2
    z = (c+(vector_type)CurrentStep+(vector_type)Device_Table_Index) % (vector_type)(256*256*256);
#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b3).s0 = (uint)charset[(z.s0 % 256) + charset_offset];
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b3).s1 = (uint)charset[(z.s1 % 256) + charset_offset];
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b3).s2 = (uint)charset[(z.s2 % 256) + charset_offset];
    (*b3).s3 = (uint)charset[(z.s3 % 256) + charset_offset];
#endif
#if grt_vector_8 || grt_vector_16
    (*b3).s4 = (uint)charset[(z.s4 % 256) + charset_offset];
    (*b3).s5 = (uint)charset[(z.s5 % 256) + charset_offset];
    (*b3).s6 = (uint)charset[(z.s6 % 256) + charset_offset];
    (*b3).s7 = (uint)charset[(z.s7 % 256) + charset_offset];
#endif
#if grt_vector_16
    (*b3).s8 = (uint)charset[(z.s8 % 256) + charset_offset];
    (*b3).s9 = (uint)charset[(z.s9 % 256) + charset_offset];
    (*b3).sA = (uint)charset[(z.sA % 256) + charset_offset];
    (*b3).sB = (uint)charset[(z.sB % 256) + charset_offset];
    (*b3).sC = (uint)charset[(z.sC % 256) + charset_offset];
    (*b3).sD = (uint)charset[(z.sD % 256) + charset_offset];
    (*b3).sE = (uint)charset[(z.sE % 256) + charset_offset];
    (*b3).sF = (uint)charset[(z.sF % 256) + charset_offset];
#endif
    if (PasswordLength == 7) {return;}

    z /= (vector_type)256;
#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b3).s0 |= (uint)charset[(z.s0 % 256) + charset_offset] << 16;
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b3).s1 |= (uint)charset[(z.s1 % 256) + charset_offset] << 16;
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b3).s2 |= (uint)charset[(z.s2 % 256) + charset_offset] << 16;
    (*b3).s3 |= (uint)charset[(z.s3 % 256) + charset_offset] << 16;
#endif
#if grt_vector_8 || grt_vector_16
    (*b3).s4 |= (uint)charset[(z.s4 % 256) + charset_offset] << 16;
    (*b3).s5 |= (uint)charset[(z.s5 % 256) + charset_offset] << 16;
    (*b3).s6 |= (uint)charset[(z.s6 % 256) + charset_offset] << 16;
    (*b3).s7 |= (uint)charset[(z.s7 % 256) + charset_offset] << 16;
#endif
#if grt_vector_16
    (*b3).s8 |= (uint)charset[(z.s8 % 256) + charset_offset] << 16;
    (*b3).s9 |= (uint)charset[(z.s9 % 256) + charset_offset] << 16;
    (*b3).sA |= (uint)charset[(z.sA % 256) + charset_offset] << 16;
    (*b3).sB |= (uint)charset[(z.sB % 256) + charset_offset] << 16;
    (*b3).sC |= (uint)charset[(z.sC % 256) + charset_offset] << 16;
    (*b3).sD |= (uint)charset[(z.sD % 256) + charset_offset] << 16;
    (*b3).sE |= (uint)charset[(z.sE % 256) + charset_offset] << 16;
    (*b3).sF |= (uint)charset[(z.sF % 256) + charset_offset] << 16;
#endif
    if (PasswordLength == 8) {return;}

    z /= (vector_type)256;
#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b4).s0 = (uint)charset[(z.s0 % 256) + charset_offset];
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b4).s1 = (uint)charset[(z.s1 % 256) + charset_offset];
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b4).s2 = (uint)charset[(z.s2 % 256) + charset_offset];
    (*b4).s3 = (uint)charset[(z.s3 % 256) + charset_offset];
#endif
#if grt_vector_8 || grt_vector_16
    (*b4).s4 = (uint)charset[(z.s4 % 256) + charset_offset];
    (*b4).s5 = (uint)charset[(z.s5 % 256) + charset_offset];
    (*b4).s6 = (uint)charset[(z.s6 % 256) + charset_offset];
    (*b4).s7 = (uint)charset[(z.s7 % 256) + charset_offset];
#endif
#if grt_vector_16
    (*b4).s8 = (uint)charset[(z.s8 % 256) + charset_offset];
    (*b4).s9 = (uint)charset[(z.s9 % 256) + charset_offset];
    (*b4).sA = (uint)charset[(z.sA % 256) + charset_offset];
    (*b4).sB = (uint)charset[(z.sB % 256) + charset_offset];
    (*b4).sC = (uint)charset[(z.sC % 256) + charset_offset];
    (*b4).sD = (uint)charset[(z.sD % 256) + charset_offset];
    (*b4).sE = (uint)charset[(z.sE % 256) + charset_offset];
    (*b4).sF = (uint)charset[(z.sF % 256) + charset_offset];
#endif
    if (PasswordLength == 9) {return;}

    z = (d+(vector_type)CurrentStep+(vector_type)Device_Table_Index) % (vector_type)(256*256*256);
#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b4).s0 |= (uint)charset[(z.s0 % 256) + charset_offset] << 16;
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b4).s1 |= (uint)charset[(z.s1 % 256) + charset_offset] << 16;
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b4).s2 |= (uint)charset[(z.s2 % 256) + charset_offset] << 16;
    (*b4).s3 |= (uint)charset[(z.s3 % 256) + charset_offset] << 16;
#endif
#if grt_vector_8 || grt_vector_16
    (*b4).s4 |= (uint)charset[(z.s4 % 256) + charset_offset] << 16;
    (*b4).s5 |= (uint)charset[(z.s5 % 256) + charset_offset] << 16;
    (*b4).s6 |= (uint)charset[(z.s6 % 256) + charset_offset] << 16;
    (*b4).s7 |= (uint)charset[(z.s7 % 256) + charset_offset] << 16;
#endif
#if grt_vector_16
    (*b4).s8 |= (uint)charset[(z.s8 % 256) + charset_offset] << 16;
    (*b4).s9 |= (uint)charset[(z.s9 % 256) + charset_offset] << 16;
    (*b4).sA |= (uint)charset[(z.sA % 256) + charset_offset] << 16;
    (*b4).sB |= (uint)charset[(z.sB % 256) + charset_offset] << 16;
    (*b4).sC |= (uint)charset[(z.sC % 256) + charset_offset] << 16;
    (*b4).sD |= (uint)charset[(z.sD % 256) + charset_offset] << 16;
    (*b4).sE |= (uint)charset[(z.sE % 256) + charset_offset] << 16;
    (*b4).sF |= (uint)charset[(z.sF % 256) + charset_offset] << 16;
#endif
    if (PasswordLength == 10) {return;}

    z /= (vector_type)256;
#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b5).s0 = (uint)charset[(z.s0 % 256) + charset_offset];
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b5).s1 = (uint)charset[(z.s1 % 256) + charset_offset];
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b5).s2 = (uint)charset[(z.s2 % 256) + charset_offset];
    (*b5).s3 = (uint)charset[(z.s3 % 256) + charset_offset];
#endif
#if grt_vector_8 || grt_vector_16
    (*b5).s4 = (uint)charset[(z.s4 % 256) + charset_offset];
    (*b5).s5 = (uint)charset[(z.s5 % 256) + charset_offset];
    (*b5).s6 = (uint)charset[(z.s6 % 256) + charset_offset];
    (*b5).s7 = (uint)charset[(z.s7 % 256) + charset_offset];
#endif
#if grt_vector_16
    (*b5).s8 = (uint)charset[(z.s8 % 256) + charset_offset];
    (*b5).s9 = (uint)charset[(z.s9 % 256) + charset_offset];
    (*b5).sA = (uint)charset[(z.sA % 256) + charset_offset];
    (*b5).sB = (uint)charset[(z.sB % 256) + charset_offset];
    (*b5).sC = (uint)charset[(z.sC % 256) + charset_offset];
    (*b5).sD = (uint)charset[(z.sD % 256) + charset_offset];
    (*b5).sE = (uint)charset[(z.sE % 256) + charset_offset];
    (*b5).sF = (uint)charset[(z.sF % 256) + charset_offset];
#endif
    if (PasswordLength == 11) {return;}

    z /= (vector_type)256;
#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b5).s0 |= (uint)charset[(z.s0 % 256) + charset_offset] << 16;
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b5).s1 |= (uint)charset[(z.s1 % 256) + charset_offset] << 16;
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b5).s2 |= (uint)charset[(z.s2 % 256) + charset_offset] << 16;
    (*b5).s3 |= (uint)charset[(z.s3 % 256) + charset_offset] << 16;
#endif
#if grt_vector_8 || grt_vector_16
    (*b5).s4 |= (uint)charset[(z.s4 % 256) + charset_offset] << 16;
    (*b5).s5 |= (uint)charset[(z.s5 % 256) + charset_offset] << 16;
    (*b5).s6 |= (uint)charset[(z.s6 % 256) + charset_offset] << 16;
    (*b5).s7 |= (uint)charset[(z.s7 % 256) + charset_offset] << 16;
#endif
#if grt_vector_16
    (*b5).s8 |= (uint)charset[(z.s8 % 256) + charset_offset] << 16;
    (*b5).s9 |= (uint)charset[(z.s9 % 256) + charset_offset] << 16;
    (*b5).sA |= (uint)charset[(z.sA % 256) + charset_offset] << 16;
    (*b5).sB |= (uint)charset[(z.sB % 256) + charset_offset] << 16;
    (*b5).sC |= (uint)charset[(z.sC % 256) + charset_offset] << 16;
    (*b5).sD |= (uint)charset[(z.sD % 256) + charset_offset] << 16;
    (*b5).sE |= (uint)charset[(z.sE % 256) + charset_offset] << 16;
    (*b5).sF |= (uint)charset[(z.sF % 256) + charset_offset] << 16;
#endif
    if (PasswordLength == 12) {return;}

}
