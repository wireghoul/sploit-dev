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

#define reverseMD5incrementCounters5() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p0=0;\
          p1=0;\
          p2=0;\
          p3=0;\
          p4=0;\
}}}}}}

#define reverseMD5incrementCounters6() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p0=0;\
            p1=0;\
            p2=0;\
            p3=0;\
            p4=0;\
            p5=0;\
}}}}}}}

#define reverseMD5incrementCounters7() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p0=0;\
              p1=0;\
              p2=0;\
              p3=0;\
              p4=0;\
              p5=0;\
              p6=0;\
}}}}}}}}

#define reverseMD5incrementCounters8() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p0=0;\
                p1=0;\
                p2=0;\
                p3=0;\
                p4=0;\
                p5=0;\
                p6=0;\
                p7=0;\
}}}}}}}}}

#define reverseMD5incrementCounters9() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p0=0;\
                  p1=0;\
                  p2=0;\
                  p3=0;\
                  p4=0;\
                  p5=0;\
                  p6=0;\
                  p7=0;\
                  p8=0;\
}}}}}}}}}}

#define reverseMD5incrementCounters10() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p0=0;\
                    p1=0;\
                    p2=0;\
                    p3=0;\
                    p4=0;\
                    p5=0;\
                    p6=0;\
                    p7=0;\
                    p8=0;\
                    p9=0;\
}}}}}}}}}}}

#define reverseMD5incrementCounters11() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p0=0;\
                      p1=0;\
                      p2=0;\
                      p3=0;\
                      p4=0;\
                      p5=0;\
                      p6=0;\
                      p7=0;\
                      p8=0;\
                      p9=0;\
                      p10=0;\
}}}}}}}}}}}}

#define reverseMD5incrementCounters12() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p0=0;\
                        p1=0;\
                        p2=0;\
                        p3=0;\
                        p4=0;\
                        p5=0;\
                        p6=0;\
                        p7=0;\
                        p8=0;\
                        p9=0;\
                        p10=0;\
                        p11=0;\
}}}}}}}}}}}}}

#define reverseMD5incrementCounters13() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p0=0;\
                          p1=0;\
                          p2=0;\
                          p3=0;\
                          p4=0;\
                          p5=0;\
                          p6=0;\
                          p7=0;\
                          p8=0;\
                          p9=0;\
                          p10=0;\
                          p11=0;\
                          p12=0;\
}}}}}}}}}}}}}}

#define reverseMD5incrementCounters14() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p0=0;\
                            p1=0;\
                            p2=0;\
                            p3=0;\
                            p4=0;\
                            p5=0;\
                            p6=0;\
                            p7=0;\
                            p8=0;\
                            p9=0;\
                            p10=0;\
                            p11=0;\
                            p12=0;\
                            p13=0;\
}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters15() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p0=0;\
                              p1=0;\
                              p2=0;\
                              p3=0;\
                              p4=0;\
                              p5=0;\
                              p6=0;\
                              p7=0;\
                              p8=0;\
                              p9=0;\
                              p10=0;\
                              p11=0;\
                              p12=0;\
                              p13=0;\
                              p14=0;\
}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters16() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p0=0;\
                                p1=0;\
                                p2=0;\
                                p3=0;\
                                p4=0;\
                                p5=0;\
                                p6=0;\
                                p7=0;\
                                p8=0;\
                                p9=0;\
                                p10=0;\
                                p11=0;\
                                p12=0;\
                                p13=0;\
                                p14=0;\
                                p15=0;\
}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters17() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p0=0;\
                                  p1=0;\
                                  p2=0;\
                                  p3=0;\
                                  p4=0;\
                                  p5=0;\
                                  p6=0;\
                                  p7=0;\
                                  p8=0;\
                                  p9=0;\
                                  p10=0;\
                                  p11=0;\
                                  p12=0;\
                                  p13=0;\
                                  p14=0;\
                                  p15=0;\
                                  p16=0;\
}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters18() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p0=0;\
                                    p1=0;\
                                    p2=0;\
                                    p3=0;\
                                    p4=0;\
                                    p5=0;\
                                    p6=0;\
                                    p7=0;\
                                    p8=0;\
                                    p9=0;\
                                    p10=0;\
                                    p11=0;\
                                    p12=0;\
                                    p13=0;\
                                    p14=0;\
                                    p15=0;\
                                    p16=0;\
                                    p17=0;\
}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters19() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p0=0;\
                                      p1=0;\
                                      p2=0;\
                                      p3=0;\
                                      p4=0;\
                                      p5=0;\
                                      p6=0;\
                                      p7=0;\
                                      p8=0;\
                                      p9=0;\
                                      p10=0;\
                                      p11=0;\
                                      p12=0;\
                                      p13=0;\
                                      p14=0;\
                                      p15=0;\
                                      p16=0;\
                                      p17=0;\
                                      p18=0;\
}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters20() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p0=0;\
                                        p1=0;\
                                        p2=0;\
                                        p3=0;\
                                        p4=0;\
                                        p5=0;\
                                        p6=0;\
                                        p7=0;\
                                        p8=0;\
                                        p9=0;\
                                        p10=0;\
                                        p11=0;\
                                        p12=0;\
                                        p13=0;\
                                        p14=0;\
                                        p15=0;\
                                        p16=0;\
                                        p17=0;\
                                        p18=0;\
                                        p19=0;\
}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters21() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p0=0;\
                                          p1=0;\
                                          p2=0;\
                                          p3=0;\
                                          p4=0;\
                                          p5=0;\
                                          p6=0;\
                                          p7=0;\
                                          p8=0;\
                                          p9=0;\
                                          p10=0;\
                                          p11=0;\
                                          p12=0;\
                                          p13=0;\
                                          p14=0;\
                                          p15=0;\
                                          p16=0;\
                                          p17=0;\
                                          p18=0;\
                                          p19=0;\
                                          p20=0;\
}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters22() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p0=0;\
                                            p1=0;\
                                            p2=0;\
                                            p3=0;\
                                            p4=0;\
                                            p5=0;\
                                            p6=0;\
                                            p7=0;\
                                            p8=0;\
                                            p9=0;\
                                            p10=0;\
                                            p11=0;\
                                            p12=0;\
                                            p13=0;\
                                            p14=0;\
                                            p15=0;\
                                            p16=0;\
                                            p17=0;\
                                            p18=0;\
                                            p19=0;\
                                            p20=0;\
                                            p21=0;\
}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters23() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p0=0;\
                                              p1=0;\
                                              p2=0;\
                                              p3=0;\
                                              p4=0;\
                                              p5=0;\
                                              p6=0;\
                                              p7=0;\
                                              p8=0;\
                                              p9=0;\
                                              p10=0;\
                                              p11=0;\
                                              p12=0;\
                                              p13=0;\
                                              p14=0;\
                                              p15=0;\
                                              p16=0;\
                                              p17=0;\
                                              p18=0;\
                                              p19=0;\
                                              p20=0;\
                                              p21=0;\
                                              p22=0;\
}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters24() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p0=0;\
                                                p1=0;\
                                                p2=0;\
                                                p3=0;\
                                                p4=0;\
                                                p5=0;\
                                                p6=0;\
                                                p7=0;\
                                                p8=0;\
                                                p9=0;\
                                                p10=0;\
                                                p11=0;\
                                                p12=0;\
                                                p13=0;\
                                                p14=0;\
                                                p15=0;\
                                                p16=0;\
                                                p17=0;\
                                                p18=0;\
                                                p19=0;\
                                                p20=0;\
                                                p21=0;\
                                                p22=0;\
                                                p23=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters25() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p0=0;\
                                                  p1=0;\
                                                  p2=0;\
                                                  p3=0;\
                                                  p4=0;\
                                                  p5=0;\
                                                  p6=0;\
                                                  p7=0;\
                                                  p8=0;\
                                                  p9=0;\
                                                  p10=0;\
                                                  p11=0;\
                                                  p12=0;\
                                                  p13=0;\
                                                  p14=0;\
                                                  p15=0;\
                                                  p16=0;\
                                                  p17=0;\
                                                  p18=0;\
                                                  p19=0;\
                                                  p20=0;\
                                                  p21=0;\
                                                  p22=0;\
                                                  p23=0;\
                                                  p24=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters26() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p0=0;\
                                                    p1=0;\
                                                    p2=0;\
                                                    p3=0;\
                                                    p4=0;\
                                                    p5=0;\
                                                    p6=0;\
                                                    p7=0;\
                                                    p8=0;\
                                                    p9=0;\
                                                    p10=0;\
                                                    p11=0;\
                                                    p12=0;\
                                                    p13=0;\
                                                    p14=0;\
                                                    p15=0;\
                                                    p16=0;\
                                                    p17=0;\
                                                    p18=0;\
                                                    p19=0;\
                                                    p20=0;\
                                                    p21=0;\
                                                    p22=0;\
                                                    p23=0;\
                                                    p24=0;\
                                                    p25=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters27() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p0=0;\
                                                      p1=0;\
                                                      p2=0;\
                                                      p3=0;\
                                                      p4=0;\
                                                      p5=0;\
                                                      p6=0;\
                                                      p7=0;\
                                                      p8=0;\
                                                      p9=0;\
                                                      p10=0;\
                                                      p11=0;\
                                                      p12=0;\
                                                      p13=0;\
                                                      p14=0;\
                                                      p15=0;\
                                                      p16=0;\
                                                      p17=0;\
                                                      p18=0;\
                                                      p19=0;\
                                                      p20=0;\
                                                      p21=0;\
                                                      p22=0;\
                                                      p23=0;\
                                                      p24=0;\
                                                      p25=0;\
                                                      p26=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters28() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p0=0;\
                                                        p1=0;\
                                                        p2=0;\
                                                        p3=0;\
                                                        p4=0;\
                                                        p5=0;\
                                                        p6=0;\
                                                        p7=0;\
                                                        p8=0;\
                                                        p9=0;\
                                                        p10=0;\
                                                        p11=0;\
                                                        p12=0;\
                                                        p13=0;\
                                                        p14=0;\
                                                        p15=0;\
                                                        p16=0;\
                                                        p17=0;\
                                                        p18=0;\
                                                        p19=0;\
                                                        p20=0;\
                                                        p21=0;\
                                                        p22=0;\
                                                        p23=0;\
                                                        p24=0;\
                                                        p25=0;\
                                                        p26=0;\
                                                        p27=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters29() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p0=0;\
                                                          p1=0;\
                                                          p2=0;\
                                                          p3=0;\
                                                          p4=0;\
                                                          p5=0;\
                                                          p6=0;\
                                                          p7=0;\
                                                          p8=0;\
                                                          p9=0;\
                                                          p10=0;\
                                                          p11=0;\
                                                          p12=0;\
                                                          p13=0;\
                                                          p14=0;\
                                                          p15=0;\
                                                          p16=0;\
                                                          p17=0;\
                                                          p18=0;\
                                                          p19=0;\
                                                          p20=0;\
                                                          p21=0;\
                                                          p22=0;\
                                                          p23=0;\
                                                          p24=0;\
                                                          p25=0;\
                                                          p26=0;\
                                                          p27=0;\
                                                          p28=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters30() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          p29++; \
                                                          ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p0=0;\
                                                            p1=0;\
                                                            p2=0;\
                                                            p3=0;\
                                                            p4=0;\
                                                            p5=0;\
                                                            p6=0;\
                                                            p7=0;\
                                                            p8=0;\
                                                            p9=0;\
                                                            p10=0;\
                                                            p11=0;\
                                                            p12=0;\
                                                            p13=0;\
                                                            p14=0;\
                                                            p15=0;\
                                                            p16=0;\
                                                            p17=0;\
                                                            p18=0;\
                                                            p19=0;\
                                                            p20=0;\
                                                            p21=0;\
                                                            p22=0;\
                                                            p23=0;\
                                                            p24=0;\
                                                            p25=0;\
                                                            p26=0;\
                                                            p27=0;\
                                                            p28=0;\
                                                            p29=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters31() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          p29++; \
                                                          ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            p30++; \
                                                            ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p0=0;\
                                                              p1=0;\
                                                              p2=0;\
                                                              p3=0;\
                                                              p4=0;\
                                                              p5=0;\
                                                              p6=0;\
                                                              p7=0;\
                                                              p8=0;\
                                                              p9=0;\
                                                              p10=0;\
                                                              p11=0;\
                                                              p12=0;\
                                                              p13=0;\
                                                              p14=0;\
                                                              p15=0;\
                                                              p16=0;\
                                                              p17=0;\
                                                              p18=0;\
                                                              p19=0;\
                                                              p20=0;\
                                                              p21=0;\
                                                              p22=0;\
                                                              p23=0;\
                                                              p24=0;\
                                                              p25=0;\
                                                              p26=0;\
                                                              p27=0;\
                                                              p28=0;\
                                                              p29=0;\
                                                              p30=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters32() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          p29++; \
                                                          ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            p30++; \
                                                            ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              p31++; \
                                                              ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p0=0;\
                                                                p1=0;\
                                                                p2=0;\
                                                                p3=0;\
                                                                p4=0;\
                                                                p5=0;\
                                                                p6=0;\
                                                                p7=0;\
                                                                p8=0;\
                                                                p9=0;\
                                                                p10=0;\
                                                                p11=0;\
                                                                p12=0;\
                                                                p13=0;\
                                                                p14=0;\
                                                                p15=0;\
                                                                p16=0;\
                                                                p17=0;\
                                                                p18=0;\
                                                                p19=0;\
                                                                p20=0;\
                                                                p21=0;\
                                                                p22=0;\
                                                                p23=0;\
                                                                p24=0;\
                                                                p25=0;\
                                                                p26=0;\
                                                                p27=0;\
                                                                p28=0;\
                                                                p29=0;\
                                                                p30=0;\
                                                                p31=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters33() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          p29++; \
                                                          ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            p30++; \
                                                            ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              p31++; \
                                                              ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                p32++; \
                                                                ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p0=0;\
                                                                  p1=0;\
                                                                  p2=0;\
                                                                  p3=0;\
                                                                  p4=0;\
                                                                  p5=0;\
                                                                  p6=0;\
                                                                  p7=0;\
                                                                  p8=0;\
                                                                  p9=0;\
                                                                  p10=0;\
                                                                  p11=0;\
                                                                  p12=0;\
                                                                  p13=0;\
                                                                  p14=0;\
                                                                  p15=0;\
                                                                  p16=0;\
                                                                  p17=0;\
                                                                  p18=0;\
                                                                  p19=0;\
                                                                  p20=0;\
                                                                  p21=0;\
                                                                  p22=0;\
                                                                  p23=0;\
                                                                  p24=0;\
                                                                  p25=0;\
                                                                  p26=0;\
                                                                  p27=0;\
                                                                  p28=0;\
                                                                  p29=0;\
                                                                  p30=0;\
                                                                  p31=0;\
                                                                  p32=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters34() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          p29++; \
                                                          ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            p30++; \
                                                            ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              p31++; \
                                                              ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                p32++; \
                                                                ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  p33++; \
                                                                  ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p0=0;\
                                                                    p1=0;\
                                                                    p2=0;\
                                                                    p3=0;\
                                                                    p4=0;\
                                                                    p5=0;\
                                                                    p6=0;\
                                                                    p7=0;\
                                                                    p8=0;\
                                                                    p9=0;\
                                                                    p10=0;\
                                                                    p11=0;\
                                                                    p12=0;\
                                                                    p13=0;\
                                                                    p14=0;\
                                                                    p15=0;\
                                                                    p16=0;\
                                                                    p17=0;\
                                                                    p18=0;\
                                                                    p19=0;\
                                                                    p20=0;\
                                                                    p21=0;\
                                                                    p22=0;\
                                                                    p23=0;\
                                                                    p24=0;\
                                                                    p25=0;\
                                                                    p26=0;\
                                                                    p27=0;\
                                                                    p28=0;\
                                                                    p29=0;\
                                                                    p30=0;\
                                                                    p31=0;\
                                                                    p32=0;\
                                                                    p33=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters35() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          p29++; \
                                                          ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            p30++; \
                                                            ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              p31++; \
                                                              ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                p32++; \
                                                                ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  p33++; \
                                                                  ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    p34++; \
                                                                    ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p0=0;\
                                                                      p1=0;\
                                                                      p2=0;\
                                                                      p3=0;\
                                                                      p4=0;\
                                                                      p5=0;\
                                                                      p6=0;\
                                                                      p7=0;\
                                                                      p8=0;\
                                                                      p9=0;\
                                                                      p10=0;\
                                                                      p11=0;\
                                                                      p12=0;\
                                                                      p13=0;\
                                                                      p14=0;\
                                                                      p15=0;\
                                                                      p16=0;\
                                                                      p17=0;\
                                                                      p18=0;\
                                                                      p19=0;\
                                                                      p20=0;\
                                                                      p21=0;\
                                                                      p22=0;\
                                                                      p23=0;\
                                                                      p24=0;\
                                                                      p25=0;\
                                                                      p26=0;\
                                                                      p27=0;\
                                                                      p28=0;\
                                                                      p29=0;\
                                                                      p30=0;\
                                                                      p31=0;\
                                                                      p32=0;\
                                                                      p33=0;\
                                                                      p34=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters36() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          p29++; \
                                                          ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            p30++; \
                                                            ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              p31++; \
                                                              ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                p32++; \
                                                                ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  p33++; \
                                                                  ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    p34++; \
                                                                    ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      p35++; \
                                                                      ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p0=0;\
                                                                        p1=0;\
                                                                        p2=0;\
                                                                        p3=0;\
                                                                        p4=0;\
                                                                        p5=0;\
                                                                        p6=0;\
                                                                        p7=0;\
                                                                        p8=0;\
                                                                        p9=0;\
                                                                        p10=0;\
                                                                        p11=0;\
                                                                        p12=0;\
                                                                        p13=0;\
                                                                        p14=0;\
                                                                        p15=0;\
                                                                        p16=0;\
                                                                        p17=0;\
                                                                        p18=0;\
                                                                        p19=0;\
                                                                        p20=0;\
                                                                        p21=0;\
                                                                        p22=0;\
                                                                        p23=0;\
                                                                        p24=0;\
                                                                        p25=0;\
                                                                        p26=0;\
                                                                        p27=0;\
                                                                        p28=0;\
                                                                        p29=0;\
                                                                        p30=0;\
                                                                        p31=0;\
                                                                        p32=0;\
                                                                        p33=0;\
                                                                        p34=0;\
                                                                        p35=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters37() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          p29++; \
                                                          ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            p30++; \
                                                            ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              p31++; \
                                                              ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                p32++; \
                                                                ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  p33++; \
                                                                  ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    p34++; \
                                                                    ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      p35++; \
                                                                      ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        p36++; \
                                                                        ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p0=0;\
                                                                          p1=0;\
                                                                          p2=0;\
                                                                          p3=0;\
                                                                          p4=0;\
                                                                          p5=0;\
                                                                          p6=0;\
                                                                          p7=0;\
                                                                          p8=0;\
                                                                          p9=0;\
                                                                          p10=0;\
                                                                          p11=0;\
                                                                          p12=0;\
                                                                          p13=0;\
                                                                          p14=0;\
                                                                          p15=0;\
                                                                          p16=0;\
                                                                          p17=0;\
                                                                          p18=0;\
                                                                          p19=0;\
                                                                          p20=0;\
                                                                          p21=0;\
                                                                          p22=0;\
                                                                          p23=0;\
                                                                          p24=0;\
                                                                          p25=0;\
                                                                          p26=0;\
                                                                          p27=0;\
                                                                          p28=0;\
                                                                          p29=0;\
                                                                          p30=0;\
                                                                          p31=0;\
                                                                          p32=0;\
                                                                          p33=0;\
                                                                          p34=0;\
                                                                          p35=0;\
                                                                          p36=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters38() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          p29++; \
                                                          ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            p30++; \
                                                            ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              p31++; \
                                                              ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                p32++; \
                                                                ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  p33++; \
                                                                  ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    p34++; \
                                                                    ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      p35++; \
                                                                      ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        p36++; \
                                                                        ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          p37++; \
                                                                          ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p0=0;\
                                                                            p1=0;\
                                                                            p2=0;\
                                                                            p3=0;\
                                                                            p4=0;\
                                                                            p5=0;\
                                                                            p6=0;\
                                                                            p7=0;\
                                                                            p8=0;\
                                                                            p9=0;\
                                                                            p10=0;\
                                                                            p11=0;\
                                                                            p12=0;\
                                                                            p13=0;\
                                                                            p14=0;\
                                                                            p15=0;\
                                                                            p16=0;\
                                                                            p17=0;\
                                                                            p18=0;\
                                                                            p19=0;\
                                                                            p20=0;\
                                                                            p21=0;\
                                                                            p22=0;\
                                                                            p23=0;\
                                                                            p24=0;\
                                                                            p25=0;\
                                                                            p26=0;\
                                                                            p27=0;\
                                                                            p28=0;\
                                                                            p29=0;\
                                                                            p30=0;\
                                                                            p31=0;\
                                                                            p32=0;\
                                                                            p33=0;\
                                                                            p34=0;\
                                                                            p35=0;\
                                                                            p36=0;\
                                                                            p37=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters39() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          p29++; \
                                                          ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            p30++; \
                                                            ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              p31++; \
                                                              ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                p32++; \
                                                                ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  p33++; \
                                                                  ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    p34++; \
                                                                    ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      p35++; \
                                                                      ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        p36++; \
                                                                        ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          p37++; \
                                                                          ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            p38++; \
                                                                            ResetCharacterAtPosition(sharedCharset[p38], 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              p0=0;\
                                                                              p1=0;\
                                                                              p2=0;\
                                                                              p3=0;\
                                                                              p4=0;\
                                                                              p5=0;\
                                                                              p6=0;\
                                                                              p7=0;\
                                                                              p8=0;\
                                                                              p9=0;\
                                                                              p10=0;\
                                                                              p11=0;\
                                                                              p12=0;\
                                                                              p13=0;\
                                                                              p14=0;\
                                                                              p15=0;\
                                                                              p16=0;\
                                                                              p17=0;\
                                                                              p18=0;\
                                                                              p19=0;\
                                                                              p20=0;\
                                                                              p21=0;\
                                                                              p22=0;\
                                                                              p23=0;\
                                                                              p24=0;\
                                                                              p25=0;\
                                                                              p26=0;\
                                                                              p27=0;\
                                                                              p28=0;\
                                                                              p29=0;\
                                                                              p30=0;\
                                                                              p31=0;\
                                                                              p32=0;\
                                                                              p33=0;\
                                                                              p34=0;\
                                                                              p35=0;\
                                                                              p36=0;\
                                                                              p37=0;\
                                                                              p38=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters40() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          p29++; \
                                                          ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            p30++; \
                                                            ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              p31++; \
                                                              ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                p32++; \
                                                                ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  p33++; \
                                                                  ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    p34++; \
                                                                    ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      p35++; \
                                                                      ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        p36++; \
                                                                        ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          p37++; \
                                                                          ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            p38++; \
                                                                            ResetCharacterAtPosition(sharedCharset[p38], 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              p38 = 0;\
                                                                              ResetCharacterAtPosition(sharedCharset[p38], 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                              p39++; \
                                                                              ResetCharacterAtPosition(sharedCharset[p39], 39, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                              if (p39 >= sharedLengths[39]) { \
                                                                                p0=0;\
                                                                                p1=0;\
                                                                                p2=0;\
                                                                                p3=0;\
                                                                                p4=0;\
                                                                                p5=0;\
                                                                                p6=0;\
                                                                                p7=0;\
                                                                                p8=0;\
                                                                                p9=0;\
                                                                                p10=0;\
                                                                                p11=0;\
                                                                                p12=0;\
                                                                                p13=0;\
                                                                                p14=0;\
                                                                                p15=0;\
                                                                                p16=0;\
                                                                                p17=0;\
                                                                                p18=0;\
                                                                                p19=0;\
                                                                                p20=0;\
                                                                                p21=0;\
                                                                                p22=0;\
                                                                                p23=0;\
                                                                                p24=0;\
                                                                                p25=0;\
                                                                                p26=0;\
                                                                                p27=0;\
                                                                                p28=0;\
                                                                                p29=0;\
                                                                                p30=0;\
                                                                                p31=0;\
                                                                                p32=0;\
                                                                                p33=0;\
                                                                                p34=0;\
                                                                                p35=0;\
                                                                                p36=0;\
                                                                                p37=0;\
                                                                                p38=0;\
                                                                                p39=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters41() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          p29++; \
                                                          ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            p30++; \
                                                            ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              p31++; \
                                                              ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                p32++; \
                                                                ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  p33++; \
                                                                  ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    p34++; \
                                                                    ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      p35++; \
                                                                      ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        p36++; \
                                                                        ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          p37++; \
                                                                          ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            p38++; \
                                                                            ResetCharacterAtPosition(sharedCharset[p38], 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              p38 = 0;\
                                                                              ResetCharacterAtPosition(sharedCharset[p38], 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                              p39++; \
                                                                              ResetCharacterAtPosition(sharedCharset[p39], 39, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                              if (p39 >= sharedLengths[39]) { \
                                                                                p39 = 0;\
                                                                                ResetCharacterAtPosition(sharedCharset[p39], 39, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                p40++; \
                                                                                ResetCharacterAtPosition(sharedCharset[p40], 40, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                if (p40 >= sharedLengths[40]) { \
                                                                                  p0=0;\
                                                                                  p1=0;\
                                                                                  p2=0;\
                                                                                  p3=0;\
                                                                                  p4=0;\
                                                                                  p5=0;\
                                                                                  p6=0;\
                                                                                  p7=0;\
                                                                                  p8=0;\
                                                                                  p9=0;\
                                                                                  p10=0;\
                                                                                  p11=0;\
                                                                                  p12=0;\
                                                                                  p13=0;\
                                                                                  p14=0;\
                                                                                  p15=0;\
                                                                                  p16=0;\
                                                                                  p17=0;\
                                                                                  p18=0;\
                                                                                  p19=0;\
                                                                                  p20=0;\
                                                                                  p21=0;\
                                                                                  p22=0;\
                                                                                  p23=0;\
                                                                                  p24=0;\
                                                                                  p25=0;\
                                                                                  p26=0;\
                                                                                  p27=0;\
                                                                                  p28=0;\
                                                                                  p29=0;\
                                                                                  p30=0;\
                                                                                  p31=0;\
                                                                                  p32=0;\
                                                                                  p33=0;\
                                                                                  p34=0;\
                                                                                  p35=0;\
                                                                                  p36=0;\
                                                                                  p37=0;\
                                                                                  p38=0;\
                                                                                  p39=0;\
                                                                                  p40=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters42() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          p29++; \
                                                          ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            p30++; \
                                                            ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              p31++; \
                                                              ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                p32++; \
                                                                ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  p33++; \
                                                                  ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    p34++; \
                                                                    ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      p35++; \
                                                                      ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        p36++; \
                                                                        ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          p37++; \
                                                                          ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            p38++; \
                                                                            ResetCharacterAtPosition(sharedCharset[p38], 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              p38 = 0;\
                                                                              ResetCharacterAtPosition(sharedCharset[p38], 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                              p39++; \
                                                                              ResetCharacterAtPosition(sharedCharset[p39], 39, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                              if (p39 >= sharedLengths[39]) { \
                                                                                p39 = 0;\
                                                                                ResetCharacterAtPosition(sharedCharset[p39], 39, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                p40++; \
                                                                                ResetCharacterAtPosition(sharedCharset[p40], 40, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                if (p40 >= sharedLengths[40]) { \
                                                                                  p40 = 0;\
                                                                                  ResetCharacterAtPosition(sharedCharset[p40], 40, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                  p41++; \
                                                                                  ResetCharacterAtPosition(sharedCharset[p41], 41, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                  if (p41 >= sharedLengths[41]) { \
                                                                                    p0=0;\
                                                                                    p1=0;\
                                                                                    p2=0;\
                                                                                    p3=0;\
                                                                                    p4=0;\
                                                                                    p5=0;\
                                                                                    p6=0;\
                                                                                    p7=0;\
                                                                                    p8=0;\
                                                                                    p9=0;\
                                                                                    p10=0;\
                                                                                    p11=0;\
                                                                                    p12=0;\
                                                                                    p13=0;\
                                                                                    p14=0;\
                                                                                    p15=0;\
                                                                                    p16=0;\
                                                                                    p17=0;\
                                                                                    p18=0;\
                                                                                    p19=0;\
                                                                                    p20=0;\
                                                                                    p21=0;\
                                                                                    p22=0;\
                                                                                    p23=0;\
                                                                                    p24=0;\
                                                                                    p25=0;\
                                                                                    p26=0;\
                                                                                    p27=0;\
                                                                                    p28=0;\
                                                                                    p29=0;\
                                                                                    p30=0;\
                                                                                    p31=0;\
                                                                                    p32=0;\
                                                                                    p33=0;\
                                                                                    p34=0;\
                                                                                    p35=0;\
                                                                                    p36=0;\
                                                                                    p37=0;\
                                                                                    p38=0;\
                                                                                    p39=0;\
                                                                                    p40=0;\
                                                                                    p41=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters43() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          p29++; \
                                                          ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            p30++; \
                                                            ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              p31++; \
                                                              ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                p32++; \
                                                                ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  p33++; \
                                                                  ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    p34++; \
                                                                    ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      p35++; \
                                                                      ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        p36++; \
                                                                        ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          p37++; \
                                                                          ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            p38++; \
                                                                            ResetCharacterAtPosition(sharedCharset[p38], 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              p38 = 0;\
                                                                              ResetCharacterAtPosition(sharedCharset[p38], 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                              p39++; \
                                                                              ResetCharacterAtPosition(sharedCharset[p39], 39, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                              if (p39 >= sharedLengths[39]) { \
                                                                                p39 = 0;\
                                                                                ResetCharacterAtPosition(sharedCharset[p39], 39, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                p40++; \
                                                                                ResetCharacterAtPosition(sharedCharset[p40], 40, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                if (p40 >= sharedLengths[40]) { \
                                                                                  p40 = 0;\
                                                                                  ResetCharacterAtPosition(sharedCharset[p40], 40, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                  p41++; \
                                                                                  ResetCharacterAtPosition(sharedCharset[p41], 41, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                  if (p41 >= sharedLengths[41]) { \
                                                                                    p41 = 0;\
                                                                                    ResetCharacterAtPosition(sharedCharset[p41], 41, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                    p42++; \
                                                                                    ResetCharacterAtPosition(sharedCharset[p42], 42, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                    if (p42 >= sharedLengths[42]) { \
                                                                                      p0=0;\
                                                                                      p1=0;\
                                                                                      p2=0;\
                                                                                      p3=0;\
                                                                                      p4=0;\
                                                                                      p5=0;\
                                                                                      p6=0;\
                                                                                      p7=0;\
                                                                                      p8=0;\
                                                                                      p9=0;\
                                                                                      p10=0;\
                                                                                      p11=0;\
                                                                                      p12=0;\
                                                                                      p13=0;\
                                                                                      p14=0;\
                                                                                      p15=0;\
                                                                                      p16=0;\
                                                                                      p17=0;\
                                                                                      p18=0;\
                                                                                      p19=0;\
                                                                                      p20=0;\
                                                                                      p21=0;\
                                                                                      p22=0;\
                                                                                      p23=0;\
                                                                                      p24=0;\
                                                                                      p25=0;\
                                                                                      p26=0;\
                                                                                      p27=0;\
                                                                                      p28=0;\
                                                                                      p29=0;\
                                                                                      p30=0;\
                                                                                      p31=0;\
                                                                                      p32=0;\
                                                                                      p33=0;\
                                                                                      p34=0;\
                                                                                      p35=0;\
                                                                                      p36=0;\
                                                                                      p37=0;\
                                                                                      p38=0;\
                                                                                      p39=0;\
                                                                                      p40=0;\
                                                                                      p41=0;\
                                                                                      p42=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters44() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          p29++; \
                                                          ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            p30++; \
                                                            ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              p31++; \
                                                              ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                p32++; \
                                                                ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  p33++; \
                                                                  ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    p34++; \
                                                                    ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      p35++; \
                                                                      ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        p36++; \
                                                                        ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          p37++; \
                                                                          ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            p38++; \
                                                                            ResetCharacterAtPosition(sharedCharset[p38], 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              p38 = 0;\
                                                                              ResetCharacterAtPosition(sharedCharset[p38], 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                              p39++; \
                                                                              ResetCharacterAtPosition(sharedCharset[p39], 39, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                              if (p39 >= sharedLengths[39]) { \
                                                                                p39 = 0;\
                                                                                ResetCharacterAtPosition(sharedCharset[p39], 39, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                p40++; \
                                                                                ResetCharacterAtPosition(sharedCharset[p40], 40, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                if (p40 >= sharedLengths[40]) { \
                                                                                  p40 = 0;\
                                                                                  ResetCharacterAtPosition(sharedCharset[p40], 40, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                  p41++; \
                                                                                  ResetCharacterAtPosition(sharedCharset[p41], 41, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                  if (p41 >= sharedLengths[41]) { \
                                                                                    p41 = 0;\
                                                                                    ResetCharacterAtPosition(sharedCharset[p41], 41, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                    p42++; \
                                                                                    ResetCharacterAtPosition(sharedCharset[p42], 42, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                    if (p42 >= sharedLengths[42]) { \
                                                                                      p42 = 0;\
                                                                                      ResetCharacterAtPosition(sharedCharset[p42], 42, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                      p43++; \
                                                                                      ResetCharacterAtPosition(sharedCharset[p43], 43, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                      if (p43 >= sharedLengths[43]) { \
                                                                                        p0=0;\
                                                                                        p1=0;\
                                                                                        p2=0;\
                                                                                        p3=0;\
                                                                                        p4=0;\
                                                                                        p5=0;\
                                                                                        p6=0;\
                                                                                        p7=0;\
                                                                                        p8=0;\
                                                                                        p9=0;\
                                                                                        p10=0;\
                                                                                        p11=0;\
                                                                                        p12=0;\
                                                                                        p13=0;\
                                                                                        p14=0;\
                                                                                        p15=0;\
                                                                                        p16=0;\
                                                                                        p17=0;\
                                                                                        p18=0;\
                                                                                        p19=0;\
                                                                                        p20=0;\
                                                                                        p21=0;\
                                                                                        p22=0;\
                                                                                        p23=0;\
                                                                                        p24=0;\
                                                                                        p25=0;\
                                                                                        p26=0;\
                                                                                        p27=0;\
                                                                                        p28=0;\
                                                                                        p29=0;\
                                                                                        p30=0;\
                                                                                        p31=0;\
                                                                                        p32=0;\
                                                                                        p33=0;\
                                                                                        p34=0;\
                                                                                        p35=0;\
                                                                                        p36=0;\
                                                                                        p37=0;\
                                                                                        p38=0;\
                                                                                        p39=0;\
                                                                                        p40=0;\
                                                                                        p41=0;\
                                                                                        p42=0;\
                                                                                        p43=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters45() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          p29++; \
                                                          ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            p30++; \
                                                            ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              p31++; \
                                                              ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                p32++; \
                                                                ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  p33++; \
                                                                  ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    p34++; \
                                                                    ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      p35++; \
                                                                      ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        p36++; \
                                                                        ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          p37++; \
                                                                          ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            p38++; \
                                                                            ResetCharacterAtPosition(sharedCharset[p38], 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              p38 = 0;\
                                                                              ResetCharacterAtPosition(sharedCharset[p38], 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                              p39++; \
                                                                              ResetCharacterAtPosition(sharedCharset[p39], 39, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                              if (p39 >= sharedLengths[39]) { \
                                                                                p39 = 0;\
                                                                                ResetCharacterAtPosition(sharedCharset[p39], 39, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                p40++; \
                                                                                ResetCharacterAtPosition(sharedCharset[p40], 40, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                if (p40 >= sharedLengths[40]) { \
                                                                                  p40 = 0;\
                                                                                  ResetCharacterAtPosition(sharedCharset[p40], 40, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                  p41++; \
                                                                                  ResetCharacterAtPosition(sharedCharset[p41], 41, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                  if (p41 >= sharedLengths[41]) { \
                                                                                    p41 = 0;\
                                                                                    ResetCharacterAtPosition(sharedCharset[p41], 41, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                    p42++; \
                                                                                    ResetCharacterAtPosition(sharedCharset[p42], 42, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                    if (p42 >= sharedLengths[42]) { \
                                                                                      p42 = 0;\
                                                                                      ResetCharacterAtPosition(sharedCharset[p42], 42, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                      p43++; \
                                                                                      ResetCharacterAtPosition(sharedCharset[p43], 43, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                      if (p43 >= sharedLengths[43]) { \
                                                                                        p43 = 0;\
                                                                                        ResetCharacterAtPosition(sharedCharset[p43], 43, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                        p44++; \
                                                                                        ResetCharacterAtPosition(sharedCharset[p44], 44, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                        if (p44 >= sharedLengths[44]) { \
                                                                                          p0=0;\
                                                                                          p1=0;\
                                                                                          p2=0;\
                                                                                          p3=0;\
                                                                                          p4=0;\
                                                                                          p5=0;\
                                                                                          p6=0;\
                                                                                          p7=0;\
                                                                                          p8=0;\
                                                                                          p9=0;\
                                                                                          p10=0;\
                                                                                          p11=0;\
                                                                                          p12=0;\
                                                                                          p13=0;\
                                                                                          p14=0;\
                                                                                          p15=0;\
                                                                                          p16=0;\
                                                                                          p17=0;\
                                                                                          p18=0;\
                                                                                          p19=0;\
                                                                                          p20=0;\
                                                                                          p21=0;\
                                                                                          p22=0;\
                                                                                          p23=0;\
                                                                                          p24=0;\
                                                                                          p25=0;\
                                                                                          p26=0;\
                                                                                          p27=0;\
                                                                                          p28=0;\
                                                                                          p29=0;\
                                                                                          p30=0;\
                                                                                          p31=0;\
                                                                                          p32=0;\
                                                                                          p33=0;\
                                                                                          p34=0;\
                                                                                          p35=0;\
                                                                                          p36=0;\
                                                                                          p37=0;\
                                                                                          p38=0;\
                                                                                          p39=0;\
                                                                                          p40=0;\
                                                                                          p41=0;\
                                                                                          p42=0;\
                                                                                          p43=0;\
                                                                                          p44=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters46() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          p29++; \
                                                          ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            p30++; \
                                                            ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              p31++; \
                                                              ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                p32++; \
                                                                ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  p33++; \
                                                                  ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    p34++; \
                                                                    ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      p35++; \
                                                                      ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        p36++; \
                                                                        ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          p37++; \
                                                                          ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            p38++; \
                                                                            ResetCharacterAtPosition(sharedCharset[p38], 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              p38 = 0;\
                                                                              ResetCharacterAtPosition(sharedCharset[p38], 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                              p39++; \
                                                                              ResetCharacterAtPosition(sharedCharset[p39], 39, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                              if (p39 >= sharedLengths[39]) { \
                                                                                p39 = 0;\
                                                                                ResetCharacterAtPosition(sharedCharset[p39], 39, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                p40++; \
                                                                                ResetCharacterAtPosition(sharedCharset[p40], 40, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                if (p40 >= sharedLengths[40]) { \
                                                                                  p40 = 0;\
                                                                                  ResetCharacterAtPosition(sharedCharset[p40], 40, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                  p41++; \
                                                                                  ResetCharacterAtPosition(sharedCharset[p41], 41, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                  if (p41 >= sharedLengths[41]) { \
                                                                                    p41 = 0;\
                                                                                    ResetCharacterAtPosition(sharedCharset[p41], 41, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                    p42++; \
                                                                                    ResetCharacterAtPosition(sharedCharset[p42], 42, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                    if (p42 >= sharedLengths[42]) { \
                                                                                      p42 = 0;\
                                                                                      ResetCharacterAtPosition(sharedCharset[p42], 42, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                      p43++; \
                                                                                      ResetCharacterAtPosition(sharedCharset[p43], 43, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                      if (p43 >= sharedLengths[43]) { \
                                                                                        p43 = 0;\
                                                                                        ResetCharacterAtPosition(sharedCharset[p43], 43, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                        p44++; \
                                                                                        ResetCharacterAtPosition(sharedCharset[p44], 44, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                        if (p44 >= sharedLengths[44]) { \
                                                                                          p44 = 0;\
                                                                                          ResetCharacterAtPosition(sharedCharset[p44], 44, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                          p45++; \
                                                                                          ResetCharacterAtPosition(sharedCharset[p45], 45, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                          if (p45 >= sharedLengths[45]) { \
                                                                                            p0=0;\
                                                                                            p1=0;\
                                                                                            p2=0;\
                                                                                            p3=0;\
                                                                                            p4=0;\
                                                                                            p5=0;\
                                                                                            p6=0;\
                                                                                            p7=0;\
                                                                                            p8=0;\
                                                                                            p9=0;\
                                                                                            p10=0;\
                                                                                            p11=0;\
                                                                                            p12=0;\
                                                                                            p13=0;\
                                                                                            p14=0;\
                                                                                            p15=0;\
                                                                                            p16=0;\
                                                                                            p17=0;\
                                                                                            p18=0;\
                                                                                            p19=0;\
                                                                                            p20=0;\
                                                                                            p21=0;\
                                                                                            p22=0;\
                                                                                            p23=0;\
                                                                                            p24=0;\
                                                                                            p25=0;\
                                                                                            p26=0;\
                                                                                            p27=0;\
                                                                                            p28=0;\
                                                                                            p29=0;\
                                                                                            p30=0;\
                                                                                            p31=0;\
                                                                                            p32=0;\
                                                                                            p33=0;\
                                                                                            p34=0;\
                                                                                            p35=0;\
                                                                                            p36=0;\
                                                                                            p37=0;\
                                                                                            p38=0;\
                                                                                            p39=0;\
                                                                                            p40=0;\
                                                                                            p41=0;\
                                                                                            p42=0;\
                                                                                            p43=0;\
                                                                                            p44=0;\
                                                                                            p45=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters47() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          p29++; \
                                                          ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            p30++; \
                                                            ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              p31++; \
                                                              ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                p32++; \
                                                                ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  p33++; \
                                                                  ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    p34++; \
                                                                    ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      p35++; \
                                                                      ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        p36++; \
                                                                        ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          p37++; \
                                                                          ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            p38++; \
                                                                            ResetCharacterAtPosition(sharedCharset[p38], 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              p38 = 0;\
                                                                              ResetCharacterAtPosition(sharedCharset[p38], 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                              p39++; \
                                                                              ResetCharacterAtPosition(sharedCharset[p39], 39, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                              if (p39 >= sharedLengths[39]) { \
                                                                                p39 = 0;\
                                                                                ResetCharacterAtPosition(sharedCharset[p39], 39, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                p40++; \
                                                                                ResetCharacterAtPosition(sharedCharset[p40], 40, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                if (p40 >= sharedLengths[40]) { \
                                                                                  p40 = 0;\
                                                                                  ResetCharacterAtPosition(sharedCharset[p40], 40, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                  p41++; \
                                                                                  ResetCharacterAtPosition(sharedCharset[p41], 41, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                  if (p41 >= sharedLengths[41]) { \
                                                                                    p41 = 0;\
                                                                                    ResetCharacterAtPosition(sharedCharset[p41], 41, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                    p42++; \
                                                                                    ResetCharacterAtPosition(sharedCharset[p42], 42, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                    if (p42 >= sharedLengths[42]) { \
                                                                                      p42 = 0;\
                                                                                      ResetCharacterAtPosition(sharedCharset[p42], 42, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                      p43++; \
                                                                                      ResetCharacterAtPosition(sharedCharset[p43], 43, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                      if (p43 >= sharedLengths[43]) { \
                                                                                        p43 = 0;\
                                                                                        ResetCharacterAtPosition(sharedCharset[p43], 43, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                        p44++; \
                                                                                        ResetCharacterAtPosition(sharedCharset[p44], 44, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                        if (p44 >= sharedLengths[44]) { \
                                                                                          p44 = 0;\
                                                                                          ResetCharacterAtPosition(sharedCharset[p44], 44, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                          p45++; \
                                                                                          ResetCharacterAtPosition(sharedCharset[p45], 45, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                          if (p45 >= sharedLengths[45]) { \
                                                                                            p45 = 0;\
                                                                                            ResetCharacterAtPosition(sharedCharset[p45], 45, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                            p46++; \
                                                                                            ResetCharacterAtPosition(sharedCharset[p46], 46, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                            if (p46 >= sharedLengths[46]) { \
                                                                                              p0=0;\
                                                                                              p1=0;\
                                                                                              p2=0;\
                                                                                              p3=0;\
                                                                                              p4=0;\
                                                                                              p5=0;\
                                                                                              p6=0;\
                                                                                              p7=0;\
                                                                                              p8=0;\
                                                                                              p9=0;\
                                                                                              p10=0;\
                                                                                              p11=0;\
                                                                                              p12=0;\
                                                                                              p13=0;\
                                                                                              p14=0;\
                                                                                              p15=0;\
                                                                                              p16=0;\
                                                                                              p17=0;\
                                                                                              p18=0;\
                                                                                              p19=0;\
                                                                                              p20=0;\
                                                                                              p21=0;\
                                                                                              p22=0;\
                                                                                              p23=0;\
                                                                                              p24=0;\
                                                                                              p25=0;\
                                                                                              p26=0;\
                                                                                              p27=0;\
                                                                                              p28=0;\
                                                                                              p29=0;\
                                                                                              p30=0;\
                                                                                              p31=0;\
                                                                                              p32=0;\
                                                                                              p33=0;\
                                                                                              p34=0;\
                                                                                              p35=0;\
                                                                                              p36=0;\
                                                                                              p37=0;\
                                                                                              p38=0;\
                                                                                              p39=0;\
                                                                                              p40=0;\
                                                                                              p41=0;\
                                                                                              p42=0;\
                                                                                              p43=0;\
                                                                                              p44=0;\
                                                                                              p45=0;\
                                                                                              p46=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define reverseMD5incrementCounters48() { \
p0++; \
ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  ResetCharacterAtPosition(sharedCharset[p0], 0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  p1++; \
  ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    ResetCharacterAtPosition(sharedCharset[p1], 1, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    p2++; \
    ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      ResetCharacterAtPosition(sharedCharset[p2], 2, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      p3++; \
      ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        ResetCharacterAtPosition(sharedCharset[p3], 3, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        p4++; \
        ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          ResetCharacterAtPosition(sharedCharset[p4], 4, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          MD5_Reverse(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, target_a, target_b, target_c, target_d, pass_length, DEVICE_Hashes_32);\
          p5++; \
          ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            ResetCharacterAtPosition(sharedCharset[p5], 5, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            p6++; \
            ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              ResetCharacterAtPosition(sharedCharset[p6], 6, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              p7++; \
              ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                ResetCharacterAtPosition(sharedCharset[p7], 7, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                p8++; \
                ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  ResetCharacterAtPosition(sharedCharset[p8], 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  p9++; \
                  ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    ResetCharacterAtPosition(sharedCharset[p9], 9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    p10++; \
                    ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      ResetCharacterAtPosition(sharedCharset[p10], 10, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      p11++; \
                      ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        ResetCharacterAtPosition(sharedCharset[p11], 11, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        p12++; \
                        ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          ResetCharacterAtPosition(sharedCharset[p12], 12, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          p13++; \
                          ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            ResetCharacterAtPosition(sharedCharset[p13], 13, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            p14++; \
                            ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              ResetCharacterAtPosition(sharedCharset[p14], 14, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              p15++; \
                              ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                ResetCharacterAtPosition(sharedCharset[p15], 15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                p16++; \
                                ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  ResetCharacterAtPosition(sharedCharset[p16], 16, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  p17++; \
                                  ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    ResetCharacterAtPosition(sharedCharset[p17], 17, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    p18++; \
                                    ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      ResetCharacterAtPosition(sharedCharset[p18], 18, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      p19++; \
                                      ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        ResetCharacterAtPosition(sharedCharset[p19], 19, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        p20++; \
                                        ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          ResetCharacterAtPosition(sharedCharset[p20], 20, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          p21++; \
                                          ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            ResetCharacterAtPosition(sharedCharset[p21], 21, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            p22++; \
                                            ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              ResetCharacterAtPosition(sharedCharset[p22], 22, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              p23++; \
                                              ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                ResetCharacterAtPosition(sharedCharset[p23], 23, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                p24++; \
                                                ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  ResetCharacterAtPosition(sharedCharset[p24], 24, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  p25++; \
                                                  ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    ResetCharacterAtPosition(sharedCharset[p25], 25, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    p26++; \
                                                    ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      ResetCharacterAtPosition(sharedCharset[p26], 26, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      p27++; \
                                                      ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        ResetCharacterAtPosition(sharedCharset[p27], 27, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        p28++; \
                                                        ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          ResetCharacterAtPosition(sharedCharset[p28], 28, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          p29++; \
                                                          ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            ResetCharacterAtPosition(sharedCharset[p29], 29, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            p30++; \
                                                            ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              ResetCharacterAtPosition(sharedCharset[p30], 30, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              p31++; \
                                                              ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                ResetCharacterAtPosition(sharedCharset[p31], 31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                p32++; \
                                                                ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  ResetCharacterAtPosition(sharedCharset[p32], 32, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  p33++; \
                                                                  ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    ResetCharacterAtPosition(sharedCharset[p33], 33, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    p34++; \
                                                                    ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      ResetCharacterAtPosition(sharedCharset[p34], 34, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      p35++; \
                                                                      ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        ResetCharacterAtPosition(sharedCharset[p35], 35, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        p36++; \
                                                                        ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          ResetCharacterAtPosition(sharedCharset[p36], 36, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          p37++; \
                                                                          ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            ResetCharacterAtPosition(sharedCharset[p37], 37, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            p38++; \
                                                                            ResetCharacterAtPosition(sharedCharset[p38], 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              p38 = 0;\
                                                                              ResetCharacterAtPosition(sharedCharset[p38], 38, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                              p39++; \
                                                                              ResetCharacterAtPosition(sharedCharset[p39], 39, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                              if (p39 >= sharedLengths[39]) { \
                                                                                p39 = 0;\
                                                                                ResetCharacterAtPosition(sharedCharset[p39], 39, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                p40++; \
                                                                                ResetCharacterAtPosition(sharedCharset[p40], 40, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                if (p40 >= sharedLengths[40]) { \
                                                                                  p40 = 0;\
                                                                                  ResetCharacterAtPosition(sharedCharset[p40], 40, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                  p41++; \
                                                                                  ResetCharacterAtPosition(sharedCharset[p41], 41, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                  if (p41 >= sharedLengths[41]) { \
                                                                                    p41 = 0;\
                                                                                    ResetCharacterAtPosition(sharedCharset[p41], 41, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                    p42++; \
                                                                                    ResetCharacterAtPosition(sharedCharset[p42], 42, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                    if (p42 >= sharedLengths[42]) { \
                                                                                      p42 = 0;\
                                                                                      ResetCharacterAtPosition(sharedCharset[p42], 42, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                      p43++; \
                                                                                      ResetCharacterAtPosition(sharedCharset[p43], 43, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                      if (p43 >= sharedLengths[43]) { \
                                                                                        p43 = 0;\
                                                                                        ResetCharacterAtPosition(sharedCharset[p43], 43, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                        p44++; \
                                                                                        ResetCharacterAtPosition(sharedCharset[p44], 44, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                        if (p44 >= sharedLengths[44]) { \
                                                                                          p44 = 0;\
                                                                                          ResetCharacterAtPosition(sharedCharset[p44], 44, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                          p45++; \
                                                                                          ResetCharacterAtPosition(sharedCharset[p45], 45, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                          if (p45 >= sharedLengths[45]) { \
                                                                                            p45 = 0;\
                                                                                            ResetCharacterAtPosition(sharedCharset[p45], 45, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                            p46++; \
                                                                                            ResetCharacterAtPosition(sharedCharset[p46], 46, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                            if (p46 >= sharedLengths[46]) { \
                                                                                              p46 = 0;\
                                                                                              ResetCharacterAtPosition(sharedCharset[p46], 46, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                              p47++; \
                                                                                              ResetCharacterAtPosition(sharedCharset[p47], 47, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15); \
                                                                                              if (p47 >= sharedLengths[47]) { \
                                                                                                p0=0;\
                                                                                                p1=0;\
                                                                                                p2=0;\
                                                                                                p3=0;\
                                                                                                p4=0;\
                                                                                                p5=0;\
                                                                                                p6=0;\
                                                                                                p7=0;\
                                                                                                p8=0;\
                                                                                                p9=0;\
                                                                                                p10=0;\
                                                                                                p11=0;\
                                                                                                p12=0;\
                                                                                                p13=0;\
                                                                                                p14=0;\
                                                                                                p15=0;\
                                                                                                p16=0;\
                                                                                                p17=0;\
                                                                                                p18=0;\
                                                                                                p19=0;\
                                                                                                p20=0;\
                                                                                                p21=0;\
                                                                                                p22=0;\
                                                                                                p23=0;\
                                                                                                p24=0;\
                                                                                                p25=0;\
                                                                                                p26=0;\
                                                                                                p27=0;\
                                                                                                p28=0;\
                                                                                                p29=0;\
                                                                                                p30=0;\
                                                                                                p31=0;\
                                                                                                p32=0;\
                                                                                                p33=0;\
                                                                                                p34=0;\
                                                                                                p35=0;\
                                                                                                p36=0;\
                                                                                                p37=0;\
                                                                                                p38=0;\
                                                                                                p39=0;\
                                                                                                p40=0;\
                                                                                                p41=0;\
                                                                                                p42=0;\
                                                                                                p43=0;\
                                                                                                p44=0;\
                                                                                                p45=0;\
                                                                                                p46=0;\
                                                                                                p47=0;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

