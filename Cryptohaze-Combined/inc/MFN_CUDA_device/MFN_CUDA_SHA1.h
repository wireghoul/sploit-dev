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


//#define SHA1_ROTL(x, n) (((x) << (n)) | ((x) >> (32-(n))))
//#define SHA1_ROTR(x, n) (((x) >> (n)) | ((x) << (32-(n))))

#define SHA1_ROTR(x,bits) (((x & 0xffffffff) >> bits) | (x << (32 - bits)))
#define SHA1_ROTL(x,bits) (((x & 0xffffffff) << bits) | (x >> (32 - bits)))

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
