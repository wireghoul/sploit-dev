// Stuff we should do for all the kernels.


// Things we should define in the calling code...
#define CPU_DEBUG 0
//#define BITALIGN
#define NVIDIA_HACKS 1
//#define PASSWORD_LENGTH 6

// Make my UI sane...
#ifndef VECTOR_WIDTH
    //#error "VECTOR_WIDTH must be defined for compile!"
    #define VECTOR_WIDTH 4
#endif

#ifndef PASSWORD_LENGTH
    #define PASSWORD_LENGTH 12
#endif

#if VECTOR_WIDTH == 1
    #error "Vector width 1 not supported!"
#elif VECTOR_WIDTH == 2
    #define vector_type uint2
    #define vload_type vload2
    #define vstore_type vstore2
    #define grt_vector_2 1
#elif VECTOR_WIDTH == 4
    #define vector_type uint4
    #define vload_type vload4
    #define vstore_type vstore4
    #define grt_vector_4 1
#elif VECTOR_WIDTH == 8
    #define vector_type uint8
    #define vload_type vload8
    #define vstore_type vstore8
    #define grt_vector_8 1
#elif VECTOR_WIDTH == 16
    #define vector_type uint16
    #define vload_type vload16
    #define vstore_type vstore16
    #define grt_vector_16 1
#else
    #error "Vector width not specified or invalid vector width specified!"
#endif


#ifdef CPU_DEBUG
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif


// Common functions

// Copy a single charset from constant memory into shared memory.
// Used in most OpenCL code.
void copySingleCharsetToShared(__local unsigned char *sharedCharset,
        __constant unsigned char *constantCharset);


// Pads an MD style hash and sets the length in the appropriate space.
void padMDHash(int length,
        vector_type *b0, vector_type *b1, vector_type *b2, vector_type *b3, vector_type *b4, vector_type *b5, vector_type *b6, vector_type *b7,
        vector_type *b8, vector_type *b9, vector_type *b10, vector_type *b11, vector_type *b12, vector_type *b13, vector_type *b14, vector_type *b15);

void copySingleBitmapToShared(__local unsigned char *sharedBitmap,
        __constant unsigned char *globalBitmap);


void reduceSingleCharsetNormal(vector_type *b0, vector_type *b1, vector_type *b2,
        vector_type a, vector_type b, vector_type c, vector_type d,
        uint CurrentStep, __local unsigned char *charset, uint charset_offset, int PasswordLength, uint Device_Table_Index);

void ClearB0ToB15(vector_type *b0, vector_type *b1, vector_type *b2, vector_type *b3, vector_type *b4, vector_type *b5, vector_type *b6, vector_type *b7,
        vector_type *b8, vector_type *b9, vector_type *b10, vector_type *b11, vector_type *b12, vector_type *b13, vector_type *b14, vector_type *b15);


// Function bodies


inline void copySingleCharsetToShared(__local unsigned char *sharedCharset,
        __constant unsigned char *constantCharset) {

    // 32-bit accesses are used to help coalesce memory accesses.
    uint a;

    for (a = 0; a < (512); a++) {
        sharedCharset[a] = constantCharset[a];
    }
}


inline void copySingleBitmapToShared(__local unsigned char *sharedBitmap,
        __constant unsigned char *globalBitmap) {

    // 32-bit accesses are used to help coalesce memory accesses.
    uint a;

    for (a = 0; a < (8192); a++) {
        sharedBitmap[a] = globalBitmap[a];
    }
}


inline void padMDHash(int length,
                vector_type *b0, vector_type *b1, vector_type *b2, vector_type *b3, vector_type *b4, vector_type *b5, vector_type *b6, vector_type *b7,
		vector_type *b8, vector_type *b9, vector_type *b10, vector_type *b11, vector_type *b12, vector_type *b13, vector_type *b14, vector_type *b15) {

  // Set length properly (length in bits)
  *b14 = (vector_type)(length * 8);

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




inline void reduceSingleCharsetNormal(vector_type *b0, vector_type *b1, vector_type *b2,
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
    (*b0).s0 |= (uint)charset[(z.s0 % 256) + charset_offset] << 8;
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b0).s1 |= (uint)charset[(z.s1 % 256) + charset_offset] << 8;
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b0).s2 |= (uint)charset[(z.s2 % 256) + charset_offset] << 8;
    (*b0).s3 |= (uint)charset[(z.s3 % 256) + charset_offset] << 8;
#endif
#if grt_vector_8 || grt_vector_16
    (*b0).s4 |= (uint)charset[(z.s4 % 256) + charset_offset] << 8;
    (*b0).s5 |= (uint)charset[(z.s5 % 256) + charset_offset] << 8;
    (*b0).s6 |= (uint)charset[(z.s6 % 256) + charset_offset] << 8;
    (*b0).s7 |= (uint)charset[(z.s7 % 256) + charset_offset] << 8;
#endif
#if grt_vector_16
    (*b0).s8 |= (uint)charset[(z.s8 % 256) + charset_offset] << 8;
    (*b0).s9 |= (uint)charset[(z.s9 % 256) + charset_offset] << 8;
    (*b0).sA |= (uint)charset[(z.sA % 256) + charset_offset] << 8;
    (*b0).sB |= (uint)charset[(z.sB % 256) + charset_offset] << 8;
    (*b0).sC |= (uint)charset[(z.sC % 256) + charset_offset] << 8;
    (*b0).sD |= (uint)charset[(z.sD % 256) + charset_offset] << 8;
    (*b0).sE |= (uint)charset[(z.sE % 256) + charset_offset] << 8;
    (*b0).sF |= (uint)charset[(z.sF % 256) + charset_offset] << 8;
#endif
    if (PasswordLength == 2) {return;}

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
    if (PasswordLength == 3) {return;}

    // Second 3
    z = (b+(vector_type)CurrentStep+(vector_type)Device_Table_Index) % (vector_type)(256*256*256);
#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b0).s0 |= (uint)charset[(z.s0 % 256) + charset_offset] << 24;
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b0).s1 |= (uint)charset[(z.s1 % 256) + charset_offset] << 24;
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b0).s2 |= (uint)charset[(z.s2 % 256) + charset_offset] << 24;
    (*b0).s3 |= (uint)charset[(z.s3 % 256) + charset_offset] << 24;
#endif
#if grt_vector_8 || grt_vector_16
    (*b0).s4 |= (uint)charset[(z.s4 % 256) + charset_offset] << 24;
    (*b0).s5 |= (uint)charset[(z.s5 % 256) + charset_offset] << 24;
    (*b0).s6 |= (uint)charset[(z.s6 % 256) + charset_offset] << 24;
    (*b0).s7 |= (uint)charset[(z.s7 % 256) + charset_offset] << 24;
#endif
#if grt_vector_16
    (*b0).s8 |= (uint)charset[(z.s8 % 256) + charset_offset] << 24;
    (*b0).s9 |= (uint)charset[(z.s9 % 256) + charset_offset] << 24;
    (*b0).sA |= (uint)charset[(z.sA % 256) + charset_offset] << 24;
    (*b0).sB |= (uint)charset[(z.sB % 256) + charset_offset] << 24;
    (*b0).sC |= (uint)charset[(z.sC % 256) + charset_offset] << 24;
    (*b0).sD |= (uint)charset[(z.sD % 256) + charset_offset] << 24;
    (*b0).sE |= (uint)charset[(z.sE % 256) + charset_offset] << 24;
    (*b0).sF |= (uint)charset[(z.sF % 256) + charset_offset] << 24;
#endif
    if (PasswordLength == 4) {return;}

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
    if (PasswordLength == 5) {return;}

    z /= (vector_type)256;
#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b1).s0 |= (uint)charset[(z.s0 % 256) + charset_offset] << 8;
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b1).s1 |= (uint)charset[(z.s1 % 256) + charset_offset] << 8;
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b1).s2 |= (uint)charset[(z.s2 % 256) + charset_offset] << 8;
    (*b1).s3 |= (uint)charset[(z.s3 % 256) + charset_offset] << 8;
#endif
#if grt_vector_8 || grt_vector_16
    (*b1).s4 |= (uint)charset[(z.s4 % 256) + charset_offset] << 8;
    (*b1).s5 |= (uint)charset[(z.s5 % 256) + charset_offset] << 8;
    (*b1).s6 |= (uint)charset[(z.s6 % 256) + charset_offset] << 8;
    (*b1).s7 |= (uint)charset[(z.s7 % 256) + charset_offset] << 8;
#endif
#if grt_vector_16
    (*b1).s8 |= (uint)charset[(z.s8 % 256) + charset_offset] << 8;
    (*b1).s9 |= (uint)charset[(z.s9 % 256) + charset_offset] << 8;
    (*b1).sA |= (uint)charset[(z.sA % 256) + charset_offset] << 8;
    (*b1).sB |= (uint)charset[(z.sB % 256) + charset_offset] << 8;
    (*b1).sC |= (uint)charset[(z.sC % 256) + charset_offset] << 8;
    (*b1).sD |= (uint)charset[(z.sD % 256) + charset_offset] << 8;
    (*b1).sE |= (uint)charset[(z.sE % 256) + charset_offset] << 8;
    (*b1).sF |= (uint)charset[(z.sF % 256) + charset_offset] << 8;
#endif
    if (PasswordLength == 6) {return;}

    // Last 2
    z = (c+(vector_type)CurrentStep+(vector_type)Device_Table_Index) % (vector_type)(256*256*256);
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
    if (PasswordLength == 7) {return;}

    z /= (vector_type)256;
#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b1).s0 |= (uint)charset[(z.s0 % 256) + charset_offset] << 24;
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b1).s1 |= (uint)charset[(z.s1 % 256) + charset_offset] << 24;
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b1).s2 |= (uint)charset[(z.s2 % 256) + charset_offset] << 24;
    (*b1).s3 |= (uint)charset[(z.s3 % 256) + charset_offset] << 24;
#endif
#if grt_vector_8 || grt_vector_16
    (*b1).s4 |= (uint)charset[(z.s4 % 256) + charset_offset] << 24;
    (*b1).s5 |= (uint)charset[(z.s5 % 256) + charset_offset] << 24;
    (*b1).s6 |= (uint)charset[(z.s6 % 256) + charset_offset] << 24;
    (*b1).s7 |= (uint)charset[(z.s7 % 256) + charset_offset] << 24;
#endif
#if grt_vector_16
    (*b1).s8 |= (uint)charset[(z.s8 % 256) + charset_offset] << 24;
    (*b1).s9 |= (uint)charset[(z.s9 % 256) + charset_offset] << 24;
    (*b1).sA |= (uint)charset[(z.sA % 256) + charset_offset] << 24;
    (*b1).sB |= (uint)charset[(z.sB % 256) + charset_offset] << 24;
    (*b1).sC |= (uint)charset[(z.sC % 256) + charset_offset] << 24;
    (*b1).sD |= (uint)charset[(z.sD % 256) + charset_offset] << 24;
    (*b1).sE |= (uint)charset[(z.sE % 256) + charset_offset] << 24;
    (*b1).sF |= (uint)charset[(z.sF % 256) + charset_offset] << 24;
#endif
    if (PasswordLength == 8) {return;}

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
    if (PasswordLength == 9) {return;}

    z = (d+(vector_type)CurrentStep+(vector_type)Device_Table_Index) % (vector_type)(256*256*256);
#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b2).s0 |= (uint)charset[(z.s0 % 256) + charset_offset] << 8;
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b2).s1 |= (uint)charset[(z.s1 % 256) + charset_offset] << 8;
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b2).s2 |= (uint)charset[(z.s2 % 256) + charset_offset] << 8;
    (*b2).s3 |= (uint)charset[(z.s3 % 256) + charset_offset] << 8;
#endif
#if grt_vector_8 || grt_vector_16
    (*b2).s4 |= (uint)charset[(z.s4 % 256) + charset_offset] << 8;
    (*b2).s5 |= (uint)charset[(z.s5 % 256) + charset_offset] << 8;
    (*b2).s6 |= (uint)charset[(z.s6 % 256) + charset_offset] << 8;
    (*b2).s7 |= (uint)charset[(z.s7 % 256) + charset_offset] << 8;
#endif
#if grt_vector_16
    (*b2).s8 |= (uint)charset[(z.s8 % 256) + charset_offset] << 8;
    (*b2).s9 |= (uint)charset[(z.s9 % 256) + charset_offset] << 8;
    (*b2).sA |= (uint)charset[(z.sA % 256) + charset_offset] << 8;
    (*b2).sB |= (uint)charset[(z.sB % 256) + charset_offset] << 8;
    (*b2).sC |= (uint)charset[(z.sC % 256) + charset_offset] << 8;
    (*b2).sD |= (uint)charset[(z.sD % 256) + charset_offset] << 8;
    (*b2).sE |= (uint)charset[(z.sE % 256) + charset_offset] << 8;
    (*b2).sF |= (uint)charset[(z.sF % 256) + charset_offset] << 8;
#endif
    if (PasswordLength == 10) {return;}

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
    if (PasswordLength == 11) {return;}

    z /= (vector_type)256;
#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b2).s0 |= (uint)charset[(z.s0 % 256) + charset_offset] << 24;
#endif
#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b2).s1 |= (uint)charset[(z.s1 % 256) + charset_offset] << 24;
#endif
#if grt_vector_4 || grt_vector_8 || grt_vector_16
    (*b2).s2 |= (uint)charset[(z.s2 % 256) + charset_offset] << 24;
    (*b2).s3 |= (uint)charset[(z.s3 % 256) + charset_offset] << 24;
#endif
#if grt_vector_8 || grt_vector_16
    (*b2).s4 |= (uint)charset[(z.s4 % 256) + charset_offset] << 24;
    (*b2).s5 |= (uint)charset[(z.s5 % 256) + charset_offset] << 24;
    (*b2).s6 |= (uint)charset[(z.s6 % 256) + charset_offset] << 24;
    (*b2).s7 |= (uint)charset[(z.s7 % 256) + charset_offset] << 24;
#endif
#if grt_vector_16
    (*b2).s8 |= (uint)charset[(z.s8 % 256) + charset_offset] << 24;
    (*b2).s9 |= (uint)charset[(z.s9 % 256) + charset_offset] << 24;
    (*b2).sA |= (uint)charset[(z.sA % 256) + charset_offset] << 24;
    (*b2).sB |= (uint)charset[(z.sB % 256) + charset_offset] << 24;
    (*b2).sC |= (uint)charset[(z.sC % 256) + charset_offset] << 24;
    (*b2).sD |= (uint)charset[(z.sD % 256) + charset_offset] << 24;
    (*b2).sE |= (uint)charset[(z.sE % 256) + charset_offset] << 24;
    (*b2).sF |= (uint)charset[(z.sF % 256) + charset_offset] << 24;
#endif
    if (PasswordLength == 12) {return;}
}


inline void ClearB0ToB15(vector_type *b0, vector_type *b1, vector_type *b2, vector_type *b3, vector_type *b4, vector_type *b5, vector_type *b6, vector_type *b7,
        vector_type *b8, vector_type *b9, vector_type *b10, vector_type *b11, vector_type *b12, vector_type *b13, vector_type *b14, vector_type *b15) {

    *b0 = 0x00000000;
    *b1 = 0x00000000;
    *b2 = 0x00000000;
    *b3 = 0x00000000;
    *b4 = 0x00000000;
    *b5 = 0x00000000;
    *b6 = 0x00000000;
    *b7 = 0x00000000;
    *b8 = 0x00000000;
    *b9 = 0x00000000;
    *b10 = 0x00000000;
    *b11 = 0x00000000;
    *b12 = 0x00000000;
    *b13 = 0x00000000;
    *b14 = 0x00000000;
    *b15 = 0x00000000;

}
