
// Things we should define in the calling code...
#define CPU_DEBUG 0
//#define BITALIGN 1
//#define PASSWORD_LENGTH 6



// Make my UI sane...
#ifndef __OPENCL_VERSION__
    #define __kernel
    #define __global
    #define __local
    #define __private
    #define __constant
    #define get_global_id(x)
    #define restrict
    #include <vector_types.h>
    #define NUMTHREADS 1
#endif

#ifndef VECTOR_WIDTH
    //#error "VECTOR_WIDTH must be defined for compile!"
    #define VECTOR_WIDTH 4
#endif

#ifndef PASSWORD_LENGTH
    #define PASSWORD_LENGTH 5
#endif

#if VECTOR_WIDTH == 1
    #define vector_type uint
    #define vload_type vload1
    #define vstore_type vstore1
    #define grt_vector_2 1
    #define vload1(offset, p) (offset + *p) 
    #define grt_vector_1 1
    #undef E0
    #define E0
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


#ifndef SHARED_BITMAP_SIZE
#define SHARED_BITMAP_SIZE 8
#endif

#if SHARED_BITMAP_SIZE == 8
#define SHARED_BITMAP_MASK 0x0000ffff
#elif SHARED_BITMAP_SIZE == 16
#define SHARED_BITMAP_MASK 0x0001ffff
#elif SHARED_BITMAP_SIZE == 32
#define SHARED_BITMAP_MASK 0x0003ffff
#endif

#if CPU_DEBUG
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

// Define SHA1 rotate left/right operators
#define SHA1_ROTL(val, bits) rotate(val, bits)
#define SHA1_ROTR(val, bits) rotate(val, (32 - bits))

// Various operators used.
#define SHA1_CH(x,y,z)   ((z) ^ ((x) & ((y) ^ (z))))
#define SHA1_PAR(x,y,z)  ((x) ^ (y) ^ (z))
#define SHA1_MAJ(x,y,z)  (((x) & (y)) | ((z) & ((x) ^ (y))))

#define one_cycle_ch(a,b,c,d,e,x)            \
    e += SHA1_ROTR(a,27) +               \
              SHA1_CH(b,c,d) + 0x5a827999 + x;  \
    b  = SHA1_ROTR(b, 2); 

#define one_cycle_par(a,b,c,d,e,x)            \
    e += SHA1_ROTR(a,27) +               \
              SHA1_PAR(b,c,d) + 0x6ed9eba1 + x;  \
    b  = SHA1_ROTR(b, 2); 

#define one_cycle_maj(a,b,c,d,e,x)            \
    e += SHA1_ROTR(a,27) +               \
              SHA1_MAJ(b,c,d) + 0x8f1bbcdc + x;  \
    b  = SHA1_ROTR(b, 2); 

#define one_cycle_par2(a,b,c,d,e,x)            \
    e += SHA1_ROTR(a,27) +               \
              SHA1_PAR(b,c,d) + 0xca62c1d6 + x;  \
    b  = SHA1_ROTR(b, 2); 

#define print_hash(num, vector) printf("%d: %08x %08x %08x %08x %08x\n", num, a.vector, b.vector, c.vector, d.vector, e.vector);

#define print_all_hash(num) { \
print_hash(num, s0); \
}

#define reverse(x)(x>>24)|((x<<8) & 0x00FF0000)|((x>>8) & 0x0000FF00)|(x<<24);


#define SHA1_FULL_ROUNDS() { \
a=0x67452301; \
b=0xefcdab89; \
c=0x98badcfe; \
d=0x10325476; \
e=0xc3d2e1f0; \
one_cycle_ch(a,b,c,d,e,b0); \
one_cycle_ch(e,a,b,c,d,b1); \
one_cycle_ch(d,e,a,b,c,b2); \
one_cycle_ch(c,d,e,a,b,b3); \
one_cycle_ch(b,c,d,e,a,b4); \
one_cycle_ch(a,b,c,d,e,b5); \
one_cycle_ch(e,a,b,c,d,b6); \
one_cycle_ch(d,e,a,b,c,b7); \
one_cycle_ch(c,d,e,a,b,b8); \
one_cycle_ch(b,c,d,e,a,b9); \
one_cycle_ch(a,b,c,d,e,b10); \
one_cycle_ch(e,a,b,c,d,b11); \
one_cycle_ch(d,e,a,b,c,b12); \
one_cycle_ch(c,d,e,a,b,b13); \
one_cycle_ch(b,c,d,e,a,b14); \
one_cycle_ch(a,b,c,d,e,b15); \
b0 = SHA1_ROTL((b13^b8^b2^b0), 1); \
one_cycle_ch(e,a,b,c,d,b0); \
b1 = SHA1_ROTL((b14^b9^b3^b1), 1); \
one_cycle_ch(d,e,a,b,c,b1); \
b2 = SHA1_ROTL((b15^b10^b4^b2), 1); \
one_cycle_ch(c,d,e,a,b,b2); \
b3 = SHA1_ROTL((b0^b11^b5^b3), 1); \
one_cycle_ch(b,c,d,e,a,b3); \
b4 = SHA1_ROTL((b1^b12^b6^b4), 1); \
one_cycle_par(a,b,c,d,e,b4); \
b5 = SHA1_ROTL((b2^b13^b7^b5), 1); \
one_cycle_par(e,a,b,c,d,b5); \
b6 = SHA1_ROTL((b3^b14^b8^b6), 1); \
one_cycle_par(d,e,a,b,c,b6); \
b7 = SHA1_ROTL((b4^b15^b9^b7), 1); \
one_cycle_par(c,d,e,a,b,b7); \
b8 = SHA1_ROTL((b5^b0^b10^b8), 1); \
one_cycle_par(b,c,d,e,a,b8); \
b9 = SHA1_ROTL((b6^b1^b11^b9), 1); \
one_cycle_par(a,b,c,d,e,b9); \
b10 = SHA1_ROTL((b7^b2^b12^b10), 1); \
one_cycle_par(e,a,b,c,d,b10); \
b11 = SHA1_ROTL((b8^b3^b13^b11), 1); \
one_cycle_par(d,e,a,b,c,b11); \
b12 = SHA1_ROTL((b9^b4^b14^b12), 1); \
one_cycle_par(c,d,e,a,b,b12); \
b13 = SHA1_ROTL((b10^b5^b15^b13), 1); \
one_cycle_par(b,c,d,e,a,b13); \
b14 = SHA1_ROTL((b11^b6^b0^b14), 1); \
one_cycle_par(a,b,c,d,e,b14); \
b15 = SHA1_ROTL((b12^b7^b1^b15), 1); \
one_cycle_par(e,a,b,c,d,b15); \
b0 = SHA1_ROTL((b13^b8^b2^b0), 1); \
one_cycle_par(d,e,a,b,c,b0); \
b1 = SHA1_ROTL((b14^b9^b3^b1), 1); \
one_cycle_par(c,d,e,a,b,b1); \
b2 = SHA1_ROTL((b15^b10^b4^b2), 1); \
one_cycle_par(b,c,d,e,a,b2); \
b3 = SHA1_ROTL((b0^b11^b5^b3), 1); \
one_cycle_par(a,b,c,d,e,b3); \
b4 = SHA1_ROTL((b1^b12^b6^b4), 1); \
one_cycle_par(e,a,b,c,d,b4); \
b5 = SHA1_ROTL((b2^b13^b7^b5), 1); \
one_cycle_par(d,e,a,b,c,b5); \
b6 = SHA1_ROTL((b3^b14^b8^b6), 1); \
one_cycle_par(c,d,e,a,b,b6); \
b7 = SHA1_ROTL((b4^b15^b9^b7), 1); \
one_cycle_par(b,c,d,e,a,b7); \
b8 = SHA1_ROTL((b5^b0^b10^b8), 1); \
one_cycle_maj(a,b,c,d,e,b8); \
b9 = SHA1_ROTL((b6^b1^b11^b9), 1); \
one_cycle_maj(e,a,b,c,d,b9); \
b10 = SHA1_ROTL((b7^b2^b12^b10), 1); \
one_cycle_maj(d,e,a,b,c,b10); \
b11 = SHA1_ROTL((b8^b3^b13^b11), 1); \
one_cycle_maj(c,d,e,a,b,b11); \
b12 = SHA1_ROTL((b9^b4^b14^b12), 1); \
one_cycle_maj(b,c,d,e,a,b12); \
b13 = SHA1_ROTL((b10^b5^b15^b13), 1); \
one_cycle_maj(a,b,c,d,e,b13); \
b14 = SHA1_ROTL((b11^b6^b0^b14), 1); \
one_cycle_maj(e,a,b,c,d,b14); \
b15 = SHA1_ROTL((b12^b7^b1^b15), 1); \
one_cycle_maj(d,e,a,b,c,b15); \
b0 = SHA1_ROTL((b13^b8^b2^b0), 1); \
one_cycle_maj(c,d,e,a,b,b0); \
b1 = SHA1_ROTL((b14^b9^b3^b1), 1); \
one_cycle_maj(b,c,d,e,a,b1); \
b2 = SHA1_ROTL((b15^b10^b4^b2), 1); \
one_cycle_maj(a,b,c,d,e,b2); \
b3 = SHA1_ROTL((b0^b11^b5^b3), 1); \
one_cycle_maj(e,a,b,c,d,b3); \
b4 = SHA1_ROTL((b1^b12^b6^b4), 1); \
one_cycle_maj(d,e,a,b,c,b4); \
b5 = SHA1_ROTL((b2^b13^b7^b5), 1); \
one_cycle_maj(c,d,e,a,b,b5); \
b6 = SHA1_ROTL((b3^b14^b8^b6), 1); \
one_cycle_maj(b,c,d,e,a,b6); \
b7 = SHA1_ROTL((b4^b15^b9^b7), 1); \
one_cycle_maj(a,b,c,d,e,b7); \
b8 = SHA1_ROTL((b5^b0^b10^b8), 1); \
one_cycle_maj(e,a,b,c,d,b8); \
b9 = SHA1_ROTL((b6^b1^b11^b9), 1); \
one_cycle_maj(d,e,a,b,c,b9); \
b10 = SHA1_ROTL((b7^b2^b12^b10), 1); \
one_cycle_maj(c,d,e,a,b,b10); \
b11 = SHA1_ROTL((b8^b3^b13^b11), 1); \
one_cycle_maj(b,c,d,e,a,b11); \
b12 = SHA1_ROTL((b9^b4^b14^b12), 1); \
one_cycle_par2(a,b,c,d,e,b12); \
b13 = SHA1_ROTL((b10^b5^b15^b13), 1); \
one_cycle_par2(e,a,b,c,d,b13); \
b14 = SHA1_ROTL((b11^b6^b0^b14), 1); \
one_cycle_par2(d,e,a,b,c,b14); \
b15 = SHA1_ROTL((b12^b7^b1^b15), 1); \
one_cycle_par2(c,d,e,a,b,b15); \
b0 = SHA1_ROTL((b13^b8^b2^b0), 1); \
one_cycle_par2(b,c,d,e,a,b0); \
b1 = SHA1_ROTL((b14^b9^b3^b1), 1); \
one_cycle_par2(a,b,c,d,e,b1); \
b2 = SHA1_ROTL((b15^b10^b4^b2), 1); \
one_cycle_par2(e,a,b,c,d,b2); \
b3 = SHA1_ROTL((b0^b11^b5^b3), 1); \
one_cycle_par2(d,e,a,b,c,b3); \
b4 = SHA1_ROTL((b1^b12^b6^b4), 1); \
one_cycle_par2(c,d,e,a,b,b4); \
b5 = SHA1_ROTL((b2^b13^b7^b5), 1); \
one_cycle_par2(b,c,d,e,a,b5); \
b6 = SHA1_ROTL((b3^b14^b8^b6), 1); \
one_cycle_par2(a,b,c,d,e,b6); \
b7 = SHA1_ROTL((b4^b15^b9^b7), 1); \
one_cycle_par2(e,a,b,c,d,b7); \
b8 = SHA1_ROTL((b5^b0^b10^b8), 1); \
one_cycle_par2(d,e,a,b,c,b8); \
b9 = SHA1_ROTL((b6^b1^b11^b9), 1); \
one_cycle_par2(c,d,e,a,b,b9); \
b10 = SHA1_ROTL((b7^b2^b12^b10), 1); \
one_cycle_par2(b,c,d,e,a,b10); \
b11 = SHA1_ROTL((b8^b3^b13^b11), 1); \
one_cycle_par2(a,b,c,d,e,b11); \
b12 = SHA1_ROTL((b9^b4^b14^b12), 1); \
one_cycle_par2(e,a,b,c,d,b12); \
b13 = SHA1_ROTL((b10^b5^b15^b13), 1); \
one_cycle_par2(d,e,a,b,c,b13); \
b14 = SHA1_ROTL((b11^b6^b0^b14), 1); \
one_cycle_par2(c,d,e,a,b,b14); \
b15 = SHA1_ROTL((b12^b7^b1^b15), 1); \
one_cycle_par2(b,c,d,e,a,b15); \
a+=0x67452301; \
b+=0xefcdab89; \
c+=0x98badcfe; \
d+=0x10325476; \
e+=0xc3d2e1f0; \
}

#define SHA1_PARTIAL_ROUNDS() { \
a=0x67452301; \
b=0xefcdab89; \
c=0x98badcfe; \
d=0x10325476; \
e=0xc3d2e1f0; \
one_cycle_ch(a,b,c,d,e,b0); \
one_cycle_ch(e,a,b,c,d,b1); \
one_cycle_ch(d,e,a,b,c,b2); \
one_cycle_ch(c,d,e,a,b,b3); \
one_cycle_ch(b,c,d,e,a,b4); \
one_cycle_ch(a,b,c,d,e,b5); \
one_cycle_ch(e,a,b,c,d,b6); \
one_cycle_ch(d,e,a,b,c,b7); \
one_cycle_ch(c,d,e,a,b,b8); \
one_cycle_ch(b,c,d,e,a,b9); \
one_cycle_ch(a,b,c,d,e,b10); \
one_cycle_ch(e,a,b,c,d,b11); \
one_cycle_ch(d,e,a,b,c,b12); \
one_cycle_ch(c,d,e,a,b,b13); \
one_cycle_ch(b,c,d,e,a,b14); \
one_cycle_ch(a,b,c,d,e,b15); \
b0 = SHA1_ROTL((b13^b8^b2^b0), 1); \
one_cycle_ch(e,a,b,c,d,b0); \
b1 = SHA1_ROTL((b14^b9^b3^b1), 1); \
one_cycle_ch(d,e,a,b,c,b1); \
b2 = SHA1_ROTL((b15^b10^b4^b2), 1); \
one_cycle_ch(c,d,e,a,b,b2); \
b3 = SHA1_ROTL((b0^b11^b5^b3), 1); \
one_cycle_ch(b,c,d,e,a,b3); \
b4 = SHA1_ROTL((b1^b12^b6^b4), 1); \
one_cycle_par(a,b,c,d,e,b4); \
b5 = SHA1_ROTL((b2^b13^b7^b5), 1); \
one_cycle_par(e,a,b,c,d,b5); \
b6 = SHA1_ROTL((b3^b14^b8^b6), 1); \
one_cycle_par(d,e,a,b,c,b6); \
b7 = SHA1_ROTL((b4^b15^b9^b7), 1); \
one_cycle_par(c,d,e,a,b,b7); \
b8 = SHA1_ROTL((b5^b0^b10^b8), 1); \
one_cycle_par(b,c,d,e,a,b8); \
b9 = SHA1_ROTL((b6^b1^b11^b9), 1); \
one_cycle_par(a,b,c,d,e,b9); \
b10 = SHA1_ROTL((b7^b2^b12^b10), 1); \
one_cycle_par(e,a,b,c,d,b10); \
b11 = SHA1_ROTL((b8^b3^b13^b11), 1); \
one_cycle_par(d,e,a,b,c,b11); \
b12 = SHA1_ROTL((b9^b4^b14^b12), 1); \
one_cycle_par(c,d,e,a,b,b12); \
b13 = SHA1_ROTL((b10^b5^b15^b13), 1); \
one_cycle_par(b,c,d,e,a,b13); \
b14 = SHA1_ROTL((b11^b6^b0^b14), 1); \
one_cycle_par(a,b,c,d,e,b14); \
b15 = SHA1_ROTL((b12^b7^b1^b15), 1); \
one_cycle_par(e,a,b,c,d,b15); \
b0 = SHA1_ROTL((b13^b8^b2^b0), 1); \
one_cycle_par(d,e,a,b,c,b0); \
b1 = SHA1_ROTL((b14^b9^b3^b1), 1); \
one_cycle_par(c,d,e,a,b,b1); \
b2 = SHA1_ROTL((b15^b10^b4^b2), 1); \
one_cycle_par(b,c,d,e,a,b2); \
b3 = SHA1_ROTL((b0^b11^b5^b3), 1); \
one_cycle_par(a,b,c,d,e,b3); \
b4 = SHA1_ROTL((b1^b12^b6^b4), 1); \
one_cycle_par(e,a,b,c,d,b4); \
b5 = SHA1_ROTL((b2^b13^b7^b5), 1); \
one_cycle_par(d,e,a,b,c,b5); \
b6 = SHA1_ROTL((b3^b14^b8^b6), 1); \
one_cycle_par(c,d,e,a,b,b6); \
b7 = SHA1_ROTL((b4^b15^b9^b7), 1); \
one_cycle_par(b,c,d,e,a,b7); \
b8 = SHA1_ROTL((b5^b0^b10^b8), 1); \
one_cycle_maj(a,b,c,d,e,b8); \
b9 = SHA1_ROTL((b6^b1^b11^b9), 1); \
one_cycle_maj(e,a,b,c,d,b9); \
b10 = SHA1_ROTL((b7^b2^b12^b10), 1); \
one_cycle_maj(d,e,a,b,c,b10); \
b11 = SHA1_ROTL((b8^b3^b13^b11), 1); \
one_cycle_maj(c,d,e,a,b,b11); \
b12 = SHA1_ROTL((b9^b4^b14^b12), 1); \
one_cycle_maj(b,c,d,e,a,b12); \
b13 = SHA1_ROTL((b10^b5^b15^b13), 1); \
one_cycle_maj(a,b,c,d,e,b13); \
b14 = SHA1_ROTL((b11^b6^b0^b14), 1); \
one_cycle_maj(e,a,b,c,d,b14); \
b15 = SHA1_ROTL((b12^b7^b1^b15), 1); \
one_cycle_maj(d,e,a,b,c,b15); \
b0 = SHA1_ROTL((b13^b8^b2^b0), 1); \
one_cycle_maj(c,d,e,a,b,b0); \
b1 = SHA1_ROTL((b14^b9^b3^b1), 1); \
one_cycle_maj(b,c,d,e,a,b1); \
b2 = SHA1_ROTL((b15^b10^b4^b2), 1); \
one_cycle_maj(a,b,c,d,e,b2); \
b3 = SHA1_ROTL((b0^b11^b5^b3), 1); \
one_cycle_maj(e,a,b,c,d,b3); \
b4 = SHA1_ROTL((b1^b12^b6^b4), 1); \
one_cycle_maj(d,e,a,b,c,b4); \
b5 = SHA1_ROTL((b2^b13^b7^b5), 1); \
one_cycle_maj(c,d,e,a,b,b5); \
b6 = SHA1_ROTL((b3^b14^b8^b6), 1); \
one_cycle_maj(b,c,d,e,a,b6); \
b7 = SHA1_ROTL((b4^b15^b9^b7), 1); \
one_cycle_maj(a,b,c,d,e,b7); \
b8 = SHA1_ROTL((b5^b0^b10^b8), 1); \
one_cycle_maj(e,a,b,c,d,b8); \
b9 = SHA1_ROTL((b6^b1^b11^b9), 1); \
one_cycle_maj(d,e,a,b,c,b9); \
b10 = SHA1_ROTL((b7^b2^b12^b10), 1); \
one_cycle_maj(c,d,e,a,b,b10); \
b11 = SHA1_ROTL((b8^b3^b13^b11), 1); \
one_cycle_maj(b,c,d,e,a,b11); \
b12 = SHA1_ROTL((b9^b4^b14^b12), 1); \
one_cycle_par2(a,b,c,d,e,b12); \
b13 = SHA1_ROTL((b10^b5^b15^b13), 1); \
one_cycle_par2(e,a,b,c,d,b13); \
b14 = SHA1_ROTL((b11^b6^b0^b14), 1); \
one_cycle_par2(d,e,a,b,c,b14); \
b15 = SHA1_ROTL((b12^b7^b1^b15), 1); \
one_cycle_par2(c,d,e,a,b,b15); \
b0 = SHA1_ROTL((b13^b8^b2^b0), 1); \
one_cycle_par2(b,c,d,e,a,b0); \
b1 = SHA1_ROTL((b14^b9^b3^b1), 1); \
one_cycle_par2(a,b,c,d,e,b1); \
b2 = SHA1_ROTL((b15^b10^b4^b2), 1); \
one_cycle_par2(e,a,b,c,d,b2); \
b3 = SHA1_ROTL((b0^b11^b5^b3), 1); \
one_cycle_par2(d,e,a,b,c,b3); \
b4 = SHA1_ROTL((b1^b12^b6^b4), 1); \
one_cycle_par2(c,d,e,a,b,b4); \
b5 = SHA1_ROTL((b2^b13^b7^b5), 1); \
one_cycle_par2(b,c,d,e,a,b5); \
b6 = SHA1_ROTL((b3^b14^b8^b6), 1); \
one_cycle_par2(a,b,c,d,e,b6); \
b7 = SHA1_ROTL((b4^b15^b9^b7), 1); \
one_cycle_par2(e,a,b,c,d,b7); \
b8 = SHA1_ROTL((b5^b0^b10^b8), 1); \
one_cycle_par2(d,e,a,b,c,b8); \
b9 = SHA1_ROTL((b6^b1^b11^b9), 1); \
one_cycle_par2(c,d,e,a,b,b9); \
b10 = SHA1_ROTL((b7^b2^b12^b10), 1); \
one_cycle_par2(b,c,d,e,a,b10); \
b11 = SHA1_ROTL((b8^b3^b13^b11), 1); \
one_cycle_par2(a,b,c,d,e,b11); \
b12 = SHA1_ROTL((b9^b4^b14^b12), 1); \
one_cycle_par2(e,a,b,c,d,b12); \
b13 = SHA1_ROTL((b10^b5^b15^b13), 1); \
one_cycle_par2(d,e,a,b,c,b13); \
b14 = SHA1_ROTL((b11^b6^b0^b14), 1); \
one_cycle_par2(c,d,e,a,b,b14); \
b15 = SHA1_ROTL((b12^b7^b1^b15), 1); \
one_cycle_par2(b,c,d,e,a,b15); \
}

/**
 * The maximum charset length supported by this hash type.
 */
#define MFN_HASH_TYPE_PLAIN_CUDA_SHA1_MAX_CHARSET_LENGTH 128

// dfp: Device Found Passwords
// dfpf: Device Found Passwords Flags
#define CopyFoundPasswordToMemory160(dfp, dfpf, suffix) { \
    switch ( PASSWORD_LENGTH ) { \
        case 16: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b3.s##suffix >> 0) & 0xff; \
        case 15: \
            dfp[search_index * PASSWORD_LENGTH + 14] = (b3.s##suffix >> 8) & 0xff; \
        case 14: \
            dfp[search_index * PASSWORD_LENGTH + 13] = (b3.s##suffix >> 16) & 0xff; \
        case 13: \
            dfp[search_index * PASSWORD_LENGTH + 12] = (b3.s##suffix >> 24) & 0xff; \
        case 12: \
            dfp[search_index * PASSWORD_LENGTH + 11] = (b2.s##suffix >> 0) & 0xff; \
        case 11: \
            dfp[search_index * PASSWORD_LENGTH + 10] = (b2.s##suffix >> 8) & 0xff; \
        case 10: \
            dfp[search_index * PASSWORD_LENGTH + 9] = (b2.s##suffix >> 16) & 0xff; \
        case 9: \
            dfp[search_index * PASSWORD_LENGTH + 8] = (b2.s##suffix >> 24) & 0xff; \
        case 8: \
            dfp[search_index * PASSWORD_LENGTH + 7] = (b1.s##suffix >> 0) & 0xff; \
        case 7: \
            dfp[search_index * PASSWORD_LENGTH + 6] = (b1.s##suffix >> 8) & 0xff; \
        case 6: \
            dfp[search_index * PASSWORD_LENGTH + 5] = (b1.s##suffix >> 16) & 0xff; \
        case 5: \
            dfp[search_index * PASSWORD_LENGTH + 4] = (b1.s##suffix >> 24) & 0xff; \
        case 4: \
            dfp[search_index * PASSWORD_LENGTH + 3] = (b0.s##suffix >> 0) & 0xff; \
        case 3: \
            dfp[search_index * PASSWORD_LENGTH + 2] = (b0.s##suffix >> 8) & 0xff; \
        case 2: \
            dfp[search_index * PASSWORD_LENGTH + 1] = (b0.s##suffix >> 16) & 0xff; \
        case 1: \
            dfp[search_index * PASSWORD_LENGTH + 0] = (b0.s##suffix >> 24) & 0xff; \
    } \
    deviceGlobalFoundPasswordFlagsPlainSHA1[search_index] = (unsigned char) 1; \
}


#define CheckPassword160(dgh, dfp, dfpf, dnh, suffix) { \
    search_high = dnh; \
    search_low = 0; \
    while (search_low < search_high) { \
        search_index = search_low + (search_high - search_low) / 2; \
        current_hash_value = dgh[5 * search_index]; \
        if (current_hash_value < a.s##suffix) { \
            search_low = search_index + 1; \
        } else { \
            search_high = search_index; \
        } \
        if ((a.s##suffix == current_hash_value) && (search_low < dnh)) { \
            break; \
        } \
    } \
    if (a.s##suffix == current_hash_value) { \
        while (search_index && (a.s##suffix == dgh[(search_index - 1) * 5])) { \
            search_index--; \
        } \
        while ((a.s##suffix == dgh[search_index * 5])) { \
            if (b.s##suffix == dgh[search_index * 5 + 1]) { \
                if (c.s##suffix == dgh[search_index * 5 + 2]) { \
                    if (d.s##suffix == dgh[search_index * 5 + 3]) { \
                    CopyFoundPasswordToMemory160(dfp, dfpf, suffix); \
                    } \
                } \
            } \
            search_index++; \
        } \
    } \
}



__kernel 
    __attribute__((vec_type_hint(vector_type)))
    __attribute__((reqd_work_group_size(THREADSPERBLOCK, 1, 1)))
void MFNHashTypePlainOpenCL_SHA1(
    __constant unsigned char const * restrict deviceCharsetPlainSHA1, /* 0 */
    __constant unsigned char const * restrict deviceReverseCharsetPlainSHA1, /* 1 */
    __constant unsigned char const * restrict charsetLengthsPlainSHA1, /* 2 */
    __constant unsigned char const * restrict constantBitmapAPlainSHA1, /* 3 */
        
    __private unsigned long const numberOfHashesPlainSHA1, /* 4 */
    __global   unsigned int const * restrict deviceGlobalHashlistAddressPlainSHA1, /* 5 */
    __global   unsigned char *deviceGlobalFoundPasswordsPlainSHA1, /* 6 */
    __global   unsigned char *deviceGlobalFoundPasswordFlagsPlainSHA1, /* 7 */
        
    __global   unsigned char const * restrict deviceGlobalBitmapAPlainSHA1, /* 8 */
    __global   unsigned char const * restrict deviceGlobalBitmapBPlainSHA1, /* 9 */
    __global   unsigned char const * restrict deviceGlobalBitmapCPlainSHA1, /* 10 */
    __global   unsigned char const * restrict deviceGlobalBitmapDPlainSHA1, /* 11 */
        
    __global   unsigned char *deviceGlobalStartPointsPlainSHA1, /* 12 */
    __private unsigned long const deviceNumberThreadsPlainSHA1, /* 13 */
    __private unsigned int const deviceNumberStepsToRunPlainSHA1, /* 14 */
    __global   unsigned int * deviceGlobalStartPasswordsPlainSHA1, /* 15 */
    __global   unsigned char const * restrict deviceGlobal256kbBitmapAPlainSHA1 /* 16 */
) {
    // Start the kernel.
    __local unsigned char sharedCharsetPlainSHA1[MFN_HASH_TYPE_PLAIN_CUDA_SHA1_MAX_CHARSET_LENGTH * PASSWORD_LENGTH];
    __local unsigned char sharedReverseCharsetPlainSHA1[MFN_HASH_TYPE_PLAIN_CUDA_SHA1_MAX_CHARSET_LENGTH * PASSWORD_LENGTH];
    __local unsigned char sharedCharsetLengthsPlainSHA1[PASSWORD_LENGTH];
    __local unsigned char sharedBitmap[SHARED_BITMAP_SIZE * 1024];
    //__local unsigned int  plainStore[(((PASSWORD_LENGTH + 1)/4) + 1) * THREADSPERBLOCK * VECTOR_WIDTH];
    
    vector_type b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a, b, c, d, e, bitmap_index;
    vector_type b0_t, b1_t, b2_t, b3_t;
    
    unsigned long password_count = 0;
    unsigned int passOffset;
    unsigned int search_high, search_low, search_index, current_hash_value;
    vector_type lookupIndex;
    vector_type passwordOffsetVector;
    vector_type newPasswordCharacters;
    vector_type lookupResult;

    if (get_local_id(0) == 0) {
        uint counter;
#pragma unroll 128
        for (counter = 0; counter < (SHARED_BITMAP_SIZE * 1024); counter++) {
            sharedBitmap[counter] = constantBitmapAPlainSHA1[counter];
        }
#pragma unroll 128
        for (counter = 0; counter < (MFN_HASH_TYPE_PLAIN_CUDA_SHA1_MAX_CHARSET_LENGTH * PASSWORD_LENGTH); counter++) {
            sharedCharsetPlainSHA1[counter] = deviceCharsetPlainSHA1[counter];
        }
#pragma unroll 128
        for (counter = 0; counter < (MFN_HASH_TYPE_PLAIN_CUDA_SHA1_MAX_CHARSET_LENGTH * PASSWORD_LENGTH); counter++) {
            sharedReverseCharsetPlainSHA1[counter] = deviceReverseCharsetPlainSHA1[counter];
        }
        for (counter = 0; counter < PASSWORD_LENGTH; counter++) {
            sharedCharsetLengthsPlainSHA1[counter] = charsetLengthsPlainSHA1[counter];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
    b15 = (vector_type) (PASSWORD_LENGTH * 8);

    a = b = c = d = e = (vector_type) 0;

    b0 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainSHA1[0]);
    if (PASSWORD_LENGTH > 3) {b1 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainSHA1[1 * deviceNumberThreadsPlainSHA1]);}
    if (PASSWORD_LENGTH > 7) {b2 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainSHA1[2 * deviceNumberThreadsPlainSHA1]);}
    if (PASSWORD_LENGTH > 11) {b3 = vload_type(get_global_id(0), &deviceGlobalStartPasswordsPlainSHA1[3 * deviceNumberThreadsPlainSHA1]);}
    
    while (password_count < deviceNumberStepsToRunPlainSHA1) {
        // Store the plains - they get mangled by SHA1 state.
//        vstore_type(b0, get_local_id(0), &plainStore[0]);
//        if (PASSWORD_LENGTH > 3) {vstore_type(b1, get_local_id(0), &plainStore[VECTOR_WIDTH * THREADSPERBLOCK]);}
//        if (PASSWORD_LENGTH > 7) {vstore_type(b2, get_local_id(0), &plainStore[2 * VECTOR_WIDTH * THREADSPERBLOCK]);}
//        if (PASSWORD_LENGTH > 11) {vstore_type(b3, get_local_id(0),&plainStore[3 * VECTOR_WIDTH * THREADSPERBLOCK]);}
    
        b0_t = b0;
        b1_t = b1; 
        b2_t = b2;
        
        b15 = (vector_type) (PASSWORD_LENGTH * 8);

        SHA1_PARTIAL_ROUNDS();

        b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = (vector_type)0;
        b0 = b0_t;
        b1 = b1_t; 
        b2 = b2_t;

//        b0 = vload_type(get_local_id(0), &plainStore[0]);
//        if (PASSWORD_LENGTH > 3) {b1 = vload_type(get_local_id(0), &plainStore[VECTOR_WIDTH * THREADSPERBLOCK]);}
//        if (PASSWORD_LENGTH > 7) {b2 = vload_type(get_local_id(0), &plainStore[2 * VECTOR_WIDTH * THREADSPERBLOCK]);}
//        if (PASSWORD_LENGTH > 11) {b3 = vload_type(get_local_id(0),&plainStore[3 * VECTOR_WIDTH * THREADSPERBLOCK]);}
//        
//        printf(".s0 pass: '%c%c%c%c%c' hash: %08x %08x %08x %08x %08x\n",
//                (b0.s0 >> 24) & 0xff, (b0.s0 >> 16) & 0xff,
//                (b0.s0 >> 8) & 0xff, (b0.s0 >> 0) & 0xff,
//                (b1.s0 >> 24) & 0xff,
//                a.s0, b.s0, c.s0, d.s0, e.s0);
//        printf(".s1 pass: '%c%c%c%c%c' hash: %08x %08x %08x %08x %08x\n",
//                (b0.s1 >> 24) & 0xff, (b0.s1 >> 16) & 0xff,
//                (b0.s1 >> 8) & 0xff, (b0.s1 >> 0) & 0xff,
//                (b1.s1 >> 24) & 0xff,
//                a.s1, b.s1, c.s1, d.s1, e.s1);
//        printf(".s2 pass: '%c%c%c%c%c' hash: %08x %08x %08x %08x %08x\n",
//                (b0.s2 >> 24) & 0xff, (b0.s2 >> 16) & 0xff,
//                (b0.s2 >> 8) & 0xff, (b0.s2 >> 0) & 0xff,
//                (b1.s2 >> 24) & 0xff,
//                a.s2, b.s2, c.s2, d.s2, e.s2);
//        printf(".s3 pass: '%c%c%c%c%c' hash: %08x %08x %08x %08x %08x\n",
//                (b0.s3 >> 24) & 0xff, (b0.s3 >> 16) & 0xff,
//                (b0.s3 >> 8) & 0xff, (b0.s3 >> 0) & 0xff,
//                (b1.s3 >> 24) & 0xff,
//                a.s3, b.s3, c.s3, d.s3, e.s3);
//        

        OpenCLPasswordCheck128_EarlyOut(sharedBitmap, deviceGlobalBitmapAPlainSHA1, 
            deviceGlobalBitmapBPlainSHA1, deviceGlobalBitmapCPlainSHA1, 
            deviceGlobalBitmapDPlainSHA1, deviceGlobalHashlistAddressPlainSHA1, 
            deviceGlobalFoundPasswordsPlainSHA1, deviceGlobalFoundPasswordFlagsPlainSHA1,
            numberOfHashesPlainSHA1, deviceGlobal256kbBitmapAPlainSHA1);

        OpenCLPasswordIncrementorBE(sharedCharsetPlainSHA1, sharedReverseCharsetPlainSHA1, sharedCharsetLengthsPlainSHA1);

        password_count++; 
    }
    vstore_type(b0, get_global_id(0), &deviceGlobalStartPasswordsPlainSHA1[0]);
    if (PASSWORD_LENGTH > 3) {vstore_type(b1, get_global_id(0), &deviceGlobalStartPasswordsPlainSHA1[1 * deviceNumberThreadsPlainSHA1]);}
    if (PASSWORD_LENGTH > 7) {vstore_type(b2, get_global_id(0), &deviceGlobalStartPasswordsPlainSHA1[2 * deviceNumberThreadsPlainSHA1]);}
    if (PASSWORD_LENGTH > 11) {vstore_type(b3, get_global_id(0), &deviceGlobalStartPasswordsPlainSHA1[3 * deviceNumberThreadsPlainSHA1]);}
  
}
