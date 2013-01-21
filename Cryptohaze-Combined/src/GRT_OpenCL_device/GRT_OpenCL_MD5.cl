
// Hash defines

#define MD5S11 7
#define MD5S12 12
#define MD5S13 17
#define MD5S14 22
#define MD5S21 5
#define MD5S22 9
#define MD5S23 14
#define MD5S24 20
#define MD5S31 4
#define MD5S32 11
#define MD5S33 16
#define MD5S34 23
#define MD5S41 6
#define MD5S42 10
#define MD5S43 15
#define MD5S44 21

// OPTIMIZED MD5 FUNCTIONS HERE
//#define MD5F(x,y,z) (((y ^ z) & x) ^ z)
//#define MD5G(x,y,z) (((x & y) & z) ^ y)

/* F, G, H and I are basic MD5 functions.
 */
#define MD5F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define MD5G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define MD5H(x, y, z) ((x) ^ (y) ^ (z))
#define MD5I(x, y, z) ((y) ^ ((x) | (~z)))


/* ROTATE_LEFT rotates x left n bits.
 */

#ifdef BITALIGN
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#define MD5ROTATE_LEFT(x, y) amd_bitalign(x, x, (uint)(32 - y))
#define MD5FF(a, b, c, d, x, s, ac) { \
 (a) += amd_bytealign((b),(c),(d)) + (x) + (vector_type)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }

#elif NVIDIA_HACKS
#define MD5ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))
#define MD5FF(a, b, c, d, x, s, ac) { \
 (a) += MD5F ((b), (c), (d)) + (x) + (vector_type)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }
#else
#define MD5ROTATE_LEFT(x, y) rotate((vector_type)x, (uint)y)
#define MD5FF(a, b, c, d, x, s, ac) { \
 (a) += MD5F ((b), (c), (d)) + (x) + (vector_type)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }
#endif


#define MD5GG(a, b, c, d, x, s, ac) { \
 (a) += MD5G ((b), (c), (d)) + (x) + (vector_type)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }
#define MD5HH(a, b, c, d, x, s, ac) { \
 (a) += MD5H ((b), (c), (d)) + (x) + (vector_type)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }
#define MD5II(a, b, c, d, x, s, ac) { \
 (a) += MD5I ((b), (c), (d)) + (x) + (vector_type)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }



// Make the Apple compiler shut up.
void OpenCL_MD5(vector_type b0, vector_type b1, vector_type b2, vector_type b3, vector_type b4, vector_type b5, vector_type b6, vector_type b7,
			   vector_type b8, vector_type b9, vector_type b10, vector_type b11, vector_type b12, vector_type b13, vector_type b14, vector_type b15,
			   vector_type *a, vector_type *b, vector_type *c, vector_type *d);


inline void OpenCL_MD5(vector_type b0, vector_type b1, vector_type b2, vector_type b3, vector_type b4, vector_type b5, vector_type b6, vector_type b7,
			   vector_type b8, vector_type b9, vector_type b10, vector_type b11, vector_type b12, vector_type b13, vector_type b14, vector_type b15,
			   vector_type *a, vector_type *b, vector_type *c, vector_type *d) {
  // Load these with vectors.
  *a = (vector_type)0x67452301;
  *b = (vector_type)0xefcdab89;
  *c = (vector_type)0x98badcfe;
  *d = (vector_type)0x10325476;

  MD5FF (*a, *b, *c, *d, b0, MD5S11, 0xd76aa478); /* 1 */
  MD5FF (*d, *a, *b, *c, b1, MD5S12, 0xe8c7b756); /* 2 */
  MD5FF (*c, *d, *a, *b, b2, MD5S13, 0x242070db); /* 3 */
  MD5FF (*b, *c, *d, *a, b3, MD5S14, 0xc1bdceee); /* 4 */
  MD5FF (*a, *b, *c, *d, b4, MD5S11, 0xf57c0faf); /* 5 */
  MD5FF (*d, *a, *b, *c, b5, MD5S12, 0x4787c62a); /* 6 */
  MD5FF (*c, *d, *a, *b, b6, MD5S13, 0xa8304613); /* 7 */
  MD5FF (*b, *c, *d, *a, b7, MD5S14, 0xfd469501); /* 8 */
  MD5FF (*a, *b, *c, *d, b8, MD5S11, 0x698098d8); /* 9 */
  MD5FF (*d, *a, *b, *c, b9, MD5S12, 0x8b44f7af); /* 10 */
  MD5FF (*c, *d, *a, *b, b10, MD5S13, 0xffff5bb1); /* 11 */
  MD5FF (*b, *c, *d, *a, b11, MD5S14, 0x895cd7be); /* 12 */
  MD5FF (*a, *b, *c, *d, b12, MD5S11, 0x6b901122); /* 13 */
  MD5FF (*d, *a, *b, *c, b13, MD5S12, 0xfd987193); /* 14 */
  MD5FF (*c, *d, *a, *b, b14, MD5S13, 0xa679438e); /* 15 */
  MD5FF (*b, *c, *d, *a, b15, MD5S14, 0x49b40821); /* 16 */

 /* Round 2 */
  MD5GG (*a, *b, *c, *d, b1, MD5S21, 0xf61e2562); /* 17 */
  MD5GG (*d, *a, *b, *c, b6, MD5S22, 0xc040b340); /* 18 */
  MD5GG (*c, *d, *a, *b, b11, MD5S23, 0x265e5a51); /* 19 */
  MD5GG (*b, *c, *d, *a, b0, MD5S24, 0xe9b6c7aa); /* 20 */
  MD5GG (*a, *b, *c, *d, b5, MD5S21, 0xd62f105d); /* 21 */
  MD5GG (*d, *a, *b, *c, b10, MD5S22,  0x2441453); /* 22 */
  MD5GG (*c, *d, *a, *b, b15, MD5S23, 0xd8a1e681); /* 23 */
  MD5GG (*b, *c, *d, *a, b4, MD5S24, 0xe7d3fbc8); /* 24 */
  MD5GG (*a, *b, *c, *d, b9, MD5S21, 0x21e1cde6); /* 25 */
  MD5GG (*d, *a, *b, *c, b14, MD5S22, 0xc33707d6); /* 26 */
  MD5GG (*c, *d, *a, *b, b3, MD5S23, 0xf4d50d87); /* 27 */
  MD5GG (*b, *c, *d, *a, b8, MD5S24, 0x455a14ed); /* 28 */
  MD5GG (*a, *b, *c, *d, b13, MD5S21, 0xa9e3e905); /* 29 */
  MD5GG (*d, *a, *b, *c, b2, MD5S22, 0xfcefa3f8); /* 30 */
  MD5GG (*c, *d, *a, *b, b7, MD5S23, 0x676f02d9); /* 31 */
  MD5GG (*b, *c, *d, *a, b12, MD5S24, 0x8d2a4c8a); /* 32 */

  /* Round 3 */
  MD5HH (*a, *b, *c, *d, b5, MD5S31, 0xfffa3942); /* 33 */
  MD5HH (*d, *a, *b, *c, b8, MD5S32, 0x8771f681); /* 34 */
  MD5HH (*c, *d, *a, *b, b11, MD5S33, 0x6d9d6122); /* 35 */
  MD5HH (*b, *c, *d, *a, b14, MD5S34, 0xfde5380c); /* 36 */
  MD5HH (*a, *b, *c, *d, b1, MD5S31, 0xa4beea44); /* 37 */
  MD5HH (*d, *a, *b, *c, b4, MD5S32, 0x4bdecfa9); /* 38 */
  MD5HH (*c, *d, *a, *b, b7, MD5S33, 0xf6bb4b60); /* 39 */
  MD5HH (*b, *c, *d, *a, b10, MD5S34, 0xbebfbc70); /* 40 */
  MD5HH (*a, *b, *c, *d, b13, MD5S31, 0x289b7ec6); /* 41 */
  MD5HH (*d, *a, *b, *c, b0, MD5S32, 0xeaa127fa); /* 42 */
  MD5HH (*c, *d, *a, *b, b3, MD5S33, 0xd4ef3085); /* 43 */
  MD5HH (*b, *c, *d, *a, b6, MD5S34,  0x4881d05); /* 44 */
  MD5HH (*a, *b, *c, *d, b9, MD5S31, 0xd9d4d039); /* 45 */
  MD5HH (*d, *a, *b, *c, b12, MD5S32, 0xe6db99e5); /* 46 */
  MD5HH (*c, *d, *a, *b, b15, MD5S33, 0x1fa27cf8); /* 47 */
  MD5HH (*b, *c, *d, *a, b2, MD5S34, 0xc4ac5665); /* 48 */

  /* Round 4 */
  MD5II (*a, *b, *c, *d, b0, MD5S41, 0xf4292244); /* 49 */
  MD5II (*d, *a, *b, *c, b7, MD5S42, 0x432aff97); /* 50 */
  MD5II (*c, *d, *a, *b, b14, MD5S43, 0xab9423a7); /* 51 */
  MD5II (*b, *c, *d, *a, b5, MD5S44, 0xfc93a039); /* 52 */
  MD5II (*a, *b, *c, *d, b12, MD5S41, 0x655b59c3); /* 53 */
  MD5II (*d, *a, *b, *c, b3, MD5S42, 0x8f0ccc92); /* 54 */
  MD5II (*c, *d, *a, *b, b10, MD5S43, 0xffeff47d); /* 55 */
  MD5II (*b, *c, *d, *a, b1, MD5S44, 0x85845dd1); /* 56 */
  MD5II (*a, *b, *c, *d, b8, MD5S41, 0x6fa87e4f); /* 57 */
  MD5II (*d, *a, *b, *c, b15, MD5S42, 0xfe2ce6e0); /* 58 */
  MD5II (*c, *d, *a, *b, b6, MD5S43, 0xa3014314); /* 59 */
  MD5II (*b, *c, *d, *a, b13, MD5S44, 0x4e0811a1); /* 60 */
  MD5II (*a, *b, *c, *d, b4, MD5S41, 0xf7537e82); /* 61 */
  MD5II (*d, *a, *b, *c, b11, MD5S42, 0xbd3af235); /* 62 */
  MD5II (*c, *d, *a, *b, b2, MD5S43, 0x2ad7d2bb); /* 63 */
  MD5II (*b, *c, *d, *a, b9, MD5S44, 0xeb86d391); /* 64 */

  // Finally, add initial values, as this is the only pass we make.
  *a += (vector_type)0x67452301;
  *b += (vector_type)0xefcdab89;
  *c += (vector_type)0x98badcfe;
  *d += (vector_type)0x10325476;
}

