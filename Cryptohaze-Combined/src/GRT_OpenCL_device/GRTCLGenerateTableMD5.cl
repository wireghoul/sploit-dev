// Kernels for MD5

/*
#ifdef __APPLE_CC__
#include <OpenCL/cl.h>
#define __kernel
#define __constant
#define __global
#define __local

#endif
*/
#define CPU_DEBUG 0

#if CPU_DEBUG
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif
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
#define MD5ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))

/* FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4.
Rotation is separate from addition to prevent recomputation.
 */
#define MD5FF(a, b, c, d, x, s, ac) { \
 (a) += MD5F ((b), (c), (d)) + (x) + (uint)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }
#define MD5GG(a, b, c, d, x, s, ac) { \
 (a) += MD5G ((b), (c), (d)) + (x) + (uint)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }
#define MD5HH(a, b, c, d, x, s, ac) { \
 (a) += MD5H ((b), (c), (d)) + (x) + (uint)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }
#define MD5II(a, b, c, d, x, s, ac) { \
 (a) += MD5I ((b), (c), (d)) + (x) + (uint)(ac); \
 (a) = MD5ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
  }


inline void OpenCL_MD5(uint b0, uint b1, uint b2, uint b3, uint b4, uint b5, uint b6, uint b7,
			   uint b8, uint b9, uint b10, uint b11, uint b12, uint b13, uint b14, uint b15,
			   uint *a, uint *b, uint *c, uint *d) {
  *a = 0x67452301;
  *b = 0xefcdab89;
  *c = 0x98badcfe;
  *d = 0x10325476;

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
  *a += 0x67452301;
  *b += 0xefcdab89;
  *c += 0x98badcfe;
  *d += 0x10325476;
}


inline void copySingleCharsetToShared(__local unsigned char *sharedCharset,
        __constant unsigned char *constantCharset) {

    // 32-bit accesses are used to help coalesce memory accesses.
    uint a;

    for (a = 0; a < (512); a++) {
        sharedCharset[a] = constantCharset[a];
    }
}


inline void padMDHash(int length,
                uint *b0, uint *b1, uint *b2, uint *b3, uint *b4, uint *b5, uint *b6, uint *b7,
		uint *b8, uint *b9, uint *b10, uint *b11, uint *b12, uint *b13, uint *b14, uint *b15) {

  // Set length properly (length in bits)
  *b14 = length * 8;

  if (length == 0) {
    *b0 |= 0x00000080;
    return;
  }
  if (length == 1) {
    *b0 |= 0x00008000;
    return;
  }
  if (length == 2) {
    *b0 |= 0x00800000;
    return;
  }
  if (length == 3) {
    *b0 |= 0x80000000;
    return;
  }
  if (length == 4) {
    *b1 |= 0x00000080;
    return;
  }
  if (length == 5) {
    *b1 |= 0x00008000;
    return;
  }
  if (length == 6) {
    *b1 |= 0x00800000;
    return;
  }
  if (length == 7) {
    *b1 |= 0x80000000;
    return;
  }
  if (length == 8) {
    *b2 |= 0x00000080;
    return;
  }
  if (length == 9) {
    *b2 |= 0x00008000;
    return;
  }
  if (length == 10) {
    *b2 |= 0x00800000;
    return;
  }
  if (length == 11) {
   *b2 |= 0x80000000;
    return;
  }
  if (length == 12) {
    *b3 |= 0x00000080;
    return;
  }
  if (length == 13) {
    *b3 |= 0x00008000;
    return;
  }
  if (length == 14) {
    *b3 |= 0x00800000;
    return;
  }
  if (length == 15) {
    *b3 |= 0x80000000;
    return;
  }
  if (length == 16) {
    *b4 |= 0x00000080;
    return;
  }
  if (length == 17) {
    *b4 |= 0x00008000;
    return;
  }
  if (length == 18) {
    *b4 |= 0x00800000;
    return;
  }
  if (length == 19) {
    *b4 |= 0x80000000;
    return;
  }
  if (length == 20) {
    *b5 |= 0x00000080;
    return;
  }
  if (length == 21) {
    *b5 |= 0x00008000;
    return;
  }
  if (length == 22) {
    *b5 |= 0x00800000;
    return;
  }
  if (length == 23) {
    *b5 |= 0x80000000;
    return;
  }
}


inline void reduceSingleCharsetNormal(uint *b0, uint *b1, uint *b2,
        uint a, uint b, uint c, uint d,
        uint CurrentStep, __local unsigned char charset[], uint charset_offset, int PasswordLength, uint Device_Table_Index) {

    uint z;
    // Reduce it
    // First 3
    z = (uint)(a+CurrentStep+Device_Table_Index) % (256*256*256);
    *b0 = (uint)charset[(z % 256) + charset_offset];
    if (PasswordLength == 1) {return;}
    z /= 256;
    *b0 |= (uint)charset[(z % 256) + charset_offset] << 8;
    if (PasswordLength == 2) {return;}
    z /= 256;
    *b0 |= (uint)charset[(z % 256) + charset_offset] << 16;
    if (PasswordLength == 3) {return;}

    // Second 3
    z = (uint)(b+CurrentStep+Device_Table_Index) % (256*256*256);
    *b0 |= (uint)charset[(z % 256) + charset_offset] << 24;
    if (PasswordLength == 4) {return;}
    z /= 256;
    *b1 = (uint)charset[(z % 256) + charset_offset];
    if (PasswordLength == 5) {return;}
    z /= 256;
    *b1 |= (uint)charset[(z % 256) + charset_offset] << 8;
    if (PasswordLength == 6) {return;}

    // Last 2
    z = (uint)(c+CurrentStep+Device_Table_Index) % (256*256*256);
    *b1 |= (uint)charset[(z % 256) + charset_offset] << 16;
    if (PasswordLength == 7) {return;}
    z /= 256;
    *b1 |= (uint)charset[(z % 256) + charset_offset] << 24;
    if (PasswordLength == 8) {return;}
    z /= 256;
    *b2 = (uint)charset[(z % 256) + charset_offset];
    if (PasswordLength == 9) {return;}

    z = (uint)(d+CurrentStep+Device_Table_Index) % (256*256*256);
    *b2 |= (uint)charset[(z % 256) + charset_offset] << 8;
    if (PasswordLength == 10) {return;}
    z /= 256;
    *b2 |= (uint)charset[(z % 256) + charset_offset] << 16;
    if (PasswordLength == 11) {return;}
    z /= 256;
    *b2 |= (uint)charset[(z % 256) + charset_offset] << 24;
    if (PasswordLength == 12) {return;}
}

__kernel __attribute__((vec_type_hint(uint))) void GenerateMD5Len6(
    __constant unsigned char *deviceCharset,
    __private unsigned int deviceCharsetLength,
    __private unsigned int deviceChainLength,
    __private unsigned int deviceNumberOfChains,
    __private unsigned int deviceTableIndex,
    __private unsigned int deviceNumberOfThreads,
    __global   unsigned int *initialPasswordArray,
    __global   unsigned int *outputHashArray,
    __private   unsigned int passwordSpaceOffset,
    __private   unsigned int startChainIndex,
    __private   unsigned int stepsToRun,
    __private   unsigned int charsetOffset
) {

    // Start the kernel.
    __local unsigned char charset[512];

#if CPU_DEBUG
    printf("\n\n\n");
    printf("Kernel start, global id %d\n", get_global_id(0));
    printf("deviceCharsetLength: %d\n", deviceCharsetLength);
    printf("deviceCharset: %c %c %c %c ...\n", deviceCharset[0], deviceCharset[1], deviceCharset[2], deviceCharset[3]);
    printf("deviceChainLength: %d\n", deviceChainLength);
    printf("deviceNumberOfChains: %d\n", deviceNumberOfChains);
    printf("deviceTableIndex: %d\n", deviceTableIndex);
    printf("deviceNumberOfThreads: %d\n", deviceNumberOfThreads);
    printf("passwordSpaceOffset: %d\n", passwordSpaceOffset);
    printf("startChainIndex: %d\n", startChainIndex);
    printf("stepsToRun: %d\n", stepsToRun);
    printf("charsetOffset: %d\n", charsetOffset);



#endif

    // Needed variables for generation
    uint CurrentStep, PassCount, password_index;

    // Hash variables
    uint b0,b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;
    uint a,b,c,d;

    // Word-width access to the arrays
    __global uint *InitialArray32;
    __global uint *OutputArray32;
    // 32-bit accesses to the hash arrays
    InitialArray32 = (__global uint *)initialPasswordArray;
    OutputArray32 = (__global uint *)outputHashArray;

    // Generic "copy charset to shared memory" function
    copySingleCharsetToShared(charset, deviceCharset);
    //printf("Charset copied... %c %c %c %c ...\n", charset[0], charset[1], charset[2], charset[3]);

    // Figure out which password we are working on.
    password_index = (get_global_id(0) + (passwordSpaceOffset * deviceNumberOfThreads));
#if CPU_DEBUG
    printf("password index: %d\n", password_index);
    printf("startChainIndex: %d\n", startChainIndex);
#endif
    // Return if this thread is working on something beyond the end of the password space
    if (password_index >= deviceNumberOfChains) {
#if CPU_DEBUG
        printf("Returning: pass_index > deviceNumberOfChains\n");
#endif
        return;
    }

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

    // Load b0/b1 out of memory
    b0 = (uint)InitialArray32[0 * deviceNumberOfChains + password_index];
    b1 = (uint)InitialArray32[1 * deviceNumberOfChains + password_index];
    b2 = (uint)InitialArray32[2 * deviceNumberOfChains + password_index];

#if CPU_DEBUG
    printf("Initial loaded password: %08x %08x\n", b0, b1);
#endif
    for (PassCount = 0; PassCount < stepsToRun; PassCount++) {
        CurrentStep = PassCount + startChainIndex;
#if CPU_DEBUG
        printf("\nChain %d, step %d\n", password_index, PassCount);
#endif
        padMDHash(PASSWORD_LENGTH, &b0, &b1, &b2, &b3, &b4, &b5, &b6, &b7, &b8, &b9, &b10, &b11, &b12, &b13, &b14, &b15);
        OpenCL_MD5(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, &a, &b, &c, &d);
#if CPU_DEBUG
        printf("\n\n\nMD5 result... %08x %08x %08x %08x\n", a, b, c, d);
#endif
        reduceSingleCharsetNormal(&b0, &b1, &b2, a, b, c, d, CurrentStep, charset, charsetOffset, PASSWORD_LENGTH, deviceTableIndex);
#if CPU_DEBUG
        printf("New password: %08x %08x\n", b0, b1);
#endif
        charsetOffset++;
        if (charsetOffset >= deviceCharsetLength) {
            charsetOffset = 0;
        }
    }
    // Done with the number of steps we need to run

    // If we are done (or have somehow overflowed), store the result
    if (CurrentStep >= (deviceChainLength - 1)) {
#if CPU_DEBUG
        printf("\nstoring output chain: %08x %08x %08x %08x\n", a, b, c, d);
#endif
        OutputArray32[0 * deviceNumberOfChains + password_index] = a;
        OutputArray32[1 * deviceNumberOfChains + password_index] = b;
        OutputArray32[2 * deviceNumberOfChains + password_index] = c;
        OutputArray32[3 * deviceNumberOfChains + password_index] = d;
    }
    // Else, store the b0/b1 values back to the initial array for the next loop
    else {
#if CPU_DEBUG
        printf("storing state: %08x %08x\n", b0, b1);
#endif
        InitialArray32[0 * deviceNumberOfChains + password_index] = b0;
        InitialArray32[1 * deviceNumberOfChains + password_index] = b1;
        InitialArray32[2 * deviceNumberOfChains + password_index] = b2;
    }
}
