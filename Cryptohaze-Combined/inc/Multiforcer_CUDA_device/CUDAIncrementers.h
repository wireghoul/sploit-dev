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

#define incrementCounters1() { \
p0++; \
if (p0 >= charsetLen) { \
  return;\
}

#define incrementCounters1Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  return;\
}}

#define incrementCounters2() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    return;\
}}

#define incrementCounters2Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    return;\
}}}

#define incrementCounters3() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      return;\
}}}

#define incrementCounters3Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      return;\
}}}}

#define incrementCounters4() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        return;\
}}}}

#define incrementCounters4Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        return;\
}}}}}

#define incrementCounters5() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          return;\
}}}}}

#define incrementCounters5Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          return;\
}}}}}}

#define incrementCounters6() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            return;\
}}}}}}

#define incrementCounters6Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            return;\
}}}}}}}

#define incrementCounters7() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              return;\
}}}}}}}

#define incrementCounters7Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              return;\
}}}}}}}}

#define incrementCounters8() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                return;\
}}}}}}}}

#define incrementCounters8Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                return;\
}}}}}}}}}

#define incrementCounters9() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  return;\
}}}}}}}}}

#define incrementCounters9Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  return;\
}}}}}}}}}}

#define incrementCounters10() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    return;\
}}}}}}}}}}

#define incrementCounters10Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    return;\
}}}}}}}}}}}

#define incrementCounters11() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      return;\
}}}}}}}}}}}

#define incrementCounters11Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      return;\
}}}}}}}}}}}}

#define incrementCounters12() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        return;\
}}}}}}}}}}}}

#define incrementCounters12Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        return;\
}}}}}}}}}}}}}

#define incrementCounters13() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          return;\
}}}}}}}}}}}}}

#define incrementCounters13Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          return;\
}}}}}}}}}}}}}}

#define incrementCounters14() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            return;\
}}}}}}}}}}}}}}

#define incrementCounters14Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            return;\
}}}}}}}}}}}}}}}

#define incrementCounters15() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              return;\
}}}}}}}}}}}}}}}

#define incrementCounters15Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              return;\
}}}}}}}}}}}}}}}}

#define incrementCounters16() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                return;\
}}}}}}}}}}}}}}}}

#define incrementCounters16Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                return;\
}}}}}}}}}}}}}}}}}

#define incrementCounters17() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  return;\
}}}}}}}}}}}}}}}}}

#define incrementCounters17Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  return;\
}}}}}}}}}}}}}}}}}}

#define incrementCounters18() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    return;\
}}}}}}}}}}}}}}}}}}

#define incrementCounters18Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    return;\
}}}}}}}}}}}}}}}}}}}

#define incrementCounters19() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      return;\
}}}}}}}}}}}}}}}}}}}

#define incrementCounters19Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      return;\
}}}}}}}}}}}}}}}}}}}}

#define incrementCounters20() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        return;\
}}}}}}}}}}}}}}}}}}}}

#define incrementCounters20Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        return;\
}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters21() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          return;\
}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters21Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          return;\
}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters22() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            return;\
}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters22Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            return;\
}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters23() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              return;\
}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters23Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              return;\
}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters24() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                return;\
}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters24Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                return;\
}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters25() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  return;\
}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters25Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters26() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters26Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters27() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters27Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters28() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters28Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters29() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters29Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters30() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= charsetLen) { \
                                                            return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters30Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters31() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= charsetLen) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= charsetLen) { \
                                                              return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters31Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters32() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= charsetLen) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= charsetLen) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= charsetLen) { \
                                                                return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters32Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters33() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= charsetLen) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= charsetLen) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= charsetLen) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= charsetLen) { \
                                                                  return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters33Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters34() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= charsetLen) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= charsetLen) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= charsetLen) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= charsetLen) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= charsetLen) { \
                                                                    return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters34Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters35() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= charsetLen) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= charsetLen) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= charsetLen) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= charsetLen) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= charsetLen) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= charsetLen) { \
                                                                      return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters35Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters36() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= charsetLen) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= charsetLen) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= charsetLen) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= charsetLen) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= charsetLen) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= charsetLen) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= charsetLen) { \
                                                                        return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters36Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters37() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= charsetLen) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= charsetLen) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= charsetLen) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= charsetLen) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= charsetLen) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= charsetLen) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= charsetLen) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= charsetLen) { \
                                                                          return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters37Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters38() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= charsetLen) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= charsetLen) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= charsetLen) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= charsetLen) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= charsetLen) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= charsetLen) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= charsetLen) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= charsetLen) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= charsetLen) { \
                                                                            return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters38Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters39() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= charsetLen) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= charsetLen) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= charsetLen) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= charsetLen) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= charsetLen) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= charsetLen) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= charsetLen) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= charsetLen) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= charsetLen) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= charsetLen) { \
                                                                              return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters39Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters40() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= charsetLen) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= charsetLen) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= charsetLen) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= charsetLen) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= charsetLen) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= charsetLen) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= charsetLen) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= charsetLen) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= charsetLen) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= charsetLen) { \
                                                                              p38 = 0;\
                                                                              p39++; \
                                                                              if (p39 >= charsetLen) { \
                                                                                return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters40Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              p38 = 0;\
                                                                              p39++; \
                                                                              if (p39 >= sharedLengths[39]) { \
                                                                                return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters41() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= charsetLen) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= charsetLen) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= charsetLen) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= charsetLen) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= charsetLen) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= charsetLen) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= charsetLen) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= charsetLen) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= charsetLen) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= charsetLen) { \
                                                                              p38 = 0;\
                                                                              p39++; \
                                                                              if (p39 >= charsetLen) { \
                                                                                p39 = 0;\
                                                                                p40++; \
                                                                                if (p40 >= charsetLen) { \
                                                                                  return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters41Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              p38 = 0;\
                                                                              p39++; \
                                                                              if (p39 >= sharedLengths[39]) { \
                                                                                p39 = 0;\
                                                                                p40++; \
                                                                                if (p40 >= sharedLengths[40]) { \
                                                                                  return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters42() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= charsetLen) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= charsetLen) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= charsetLen) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= charsetLen) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= charsetLen) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= charsetLen) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= charsetLen) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= charsetLen) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= charsetLen) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= charsetLen) { \
                                                                              p38 = 0;\
                                                                              p39++; \
                                                                              if (p39 >= charsetLen) { \
                                                                                p39 = 0;\
                                                                                p40++; \
                                                                                if (p40 >= charsetLen) { \
                                                                                  p40 = 0;\
                                                                                  p41++; \
                                                                                  if (p41 >= charsetLen) { \
                                                                                    return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters42Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              p38 = 0;\
                                                                              p39++; \
                                                                              if (p39 >= sharedLengths[39]) { \
                                                                                p39 = 0;\
                                                                                p40++; \
                                                                                if (p40 >= sharedLengths[40]) { \
                                                                                  p40 = 0;\
                                                                                  p41++; \
                                                                                  if (p41 >= sharedLengths[41]) { \
                                                                                    return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters43() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= charsetLen) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= charsetLen) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= charsetLen) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= charsetLen) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= charsetLen) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= charsetLen) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= charsetLen) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= charsetLen) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= charsetLen) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= charsetLen) { \
                                                                              p38 = 0;\
                                                                              p39++; \
                                                                              if (p39 >= charsetLen) { \
                                                                                p39 = 0;\
                                                                                p40++; \
                                                                                if (p40 >= charsetLen) { \
                                                                                  p40 = 0;\
                                                                                  p41++; \
                                                                                  if (p41 >= charsetLen) { \
                                                                                    p41 = 0;\
                                                                                    p42++; \
                                                                                    if (p42 >= charsetLen) { \
                                                                                      return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters43Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              p38 = 0;\
                                                                              p39++; \
                                                                              if (p39 >= sharedLengths[39]) { \
                                                                                p39 = 0;\
                                                                                p40++; \
                                                                                if (p40 >= sharedLengths[40]) { \
                                                                                  p40 = 0;\
                                                                                  p41++; \
                                                                                  if (p41 >= sharedLengths[41]) { \
                                                                                    p41 = 0;\
                                                                                    p42++; \
                                                                                    if (p42 >= sharedLengths[42]) { \
                                                                                      return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters44() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= charsetLen) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= charsetLen) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= charsetLen) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= charsetLen) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= charsetLen) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= charsetLen) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= charsetLen) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= charsetLen) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= charsetLen) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= charsetLen) { \
                                                                              p38 = 0;\
                                                                              p39++; \
                                                                              if (p39 >= charsetLen) { \
                                                                                p39 = 0;\
                                                                                p40++; \
                                                                                if (p40 >= charsetLen) { \
                                                                                  p40 = 0;\
                                                                                  p41++; \
                                                                                  if (p41 >= charsetLen) { \
                                                                                    p41 = 0;\
                                                                                    p42++; \
                                                                                    if (p42 >= charsetLen) { \
                                                                                      p42 = 0;\
                                                                                      p43++; \
                                                                                      if (p43 >= charsetLen) { \
                                                                                        return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters44Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              p38 = 0;\
                                                                              p39++; \
                                                                              if (p39 >= sharedLengths[39]) { \
                                                                                p39 = 0;\
                                                                                p40++; \
                                                                                if (p40 >= sharedLengths[40]) { \
                                                                                  p40 = 0;\
                                                                                  p41++; \
                                                                                  if (p41 >= sharedLengths[41]) { \
                                                                                    p41 = 0;\
                                                                                    p42++; \
                                                                                    if (p42 >= sharedLengths[42]) { \
                                                                                      p42 = 0;\
                                                                                      p43++; \
                                                                                      if (p43 >= sharedLengths[43]) { \
                                                                                        return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters45() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= charsetLen) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= charsetLen) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= charsetLen) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= charsetLen) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= charsetLen) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= charsetLen) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= charsetLen) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= charsetLen) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= charsetLen) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= charsetLen) { \
                                                                              p38 = 0;\
                                                                              p39++; \
                                                                              if (p39 >= charsetLen) { \
                                                                                p39 = 0;\
                                                                                p40++; \
                                                                                if (p40 >= charsetLen) { \
                                                                                  p40 = 0;\
                                                                                  p41++; \
                                                                                  if (p41 >= charsetLen) { \
                                                                                    p41 = 0;\
                                                                                    p42++; \
                                                                                    if (p42 >= charsetLen) { \
                                                                                      p42 = 0;\
                                                                                      p43++; \
                                                                                      if (p43 >= charsetLen) { \
                                                                                        p43 = 0;\
                                                                                        p44++; \
                                                                                        if (p44 >= charsetLen) { \
                                                                                          return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters45Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              p38 = 0;\
                                                                              p39++; \
                                                                              if (p39 >= sharedLengths[39]) { \
                                                                                p39 = 0;\
                                                                                p40++; \
                                                                                if (p40 >= sharedLengths[40]) { \
                                                                                  p40 = 0;\
                                                                                  p41++; \
                                                                                  if (p41 >= sharedLengths[41]) { \
                                                                                    p41 = 0;\
                                                                                    p42++; \
                                                                                    if (p42 >= sharedLengths[42]) { \
                                                                                      p42 = 0;\
                                                                                      p43++; \
                                                                                      if (p43 >= sharedLengths[43]) { \
                                                                                        p43 = 0;\
                                                                                        p44++; \
                                                                                        if (p44 >= sharedLengths[44]) { \
                                                                                          return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters46() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= charsetLen) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= charsetLen) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= charsetLen) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= charsetLen) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= charsetLen) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= charsetLen) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= charsetLen) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= charsetLen) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= charsetLen) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= charsetLen) { \
                                                                              p38 = 0;\
                                                                              p39++; \
                                                                              if (p39 >= charsetLen) { \
                                                                                p39 = 0;\
                                                                                p40++; \
                                                                                if (p40 >= charsetLen) { \
                                                                                  p40 = 0;\
                                                                                  p41++; \
                                                                                  if (p41 >= charsetLen) { \
                                                                                    p41 = 0;\
                                                                                    p42++; \
                                                                                    if (p42 >= charsetLen) { \
                                                                                      p42 = 0;\
                                                                                      p43++; \
                                                                                      if (p43 >= charsetLen) { \
                                                                                        p43 = 0;\
                                                                                        p44++; \
                                                                                        if (p44 >= charsetLen) { \
                                                                                          p44 = 0;\
                                                                                          p45++; \
                                                                                          if (p45 >= charsetLen) { \
                                                                                            return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters46Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              p38 = 0;\
                                                                              p39++; \
                                                                              if (p39 >= sharedLengths[39]) { \
                                                                                p39 = 0;\
                                                                                p40++; \
                                                                                if (p40 >= sharedLengths[40]) { \
                                                                                  p40 = 0;\
                                                                                  p41++; \
                                                                                  if (p41 >= sharedLengths[41]) { \
                                                                                    p41 = 0;\
                                                                                    p42++; \
                                                                                    if (p42 >= sharedLengths[42]) { \
                                                                                      p42 = 0;\
                                                                                      p43++; \
                                                                                      if (p43 >= sharedLengths[43]) { \
                                                                                        p43 = 0;\
                                                                                        p44++; \
                                                                                        if (p44 >= sharedLengths[44]) { \
                                                                                          p44 = 0;\
                                                                                          p45++; \
                                                                                          if (p45 >= sharedLengths[45]) { \
                                                                                            return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters47() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= charsetLen) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= charsetLen) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= charsetLen) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= charsetLen) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= charsetLen) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= charsetLen) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= charsetLen) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= charsetLen) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= charsetLen) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= charsetLen) { \
                                                                              p38 = 0;\
                                                                              p39++; \
                                                                              if (p39 >= charsetLen) { \
                                                                                p39 = 0;\
                                                                                p40++; \
                                                                                if (p40 >= charsetLen) { \
                                                                                  p40 = 0;\
                                                                                  p41++; \
                                                                                  if (p41 >= charsetLen) { \
                                                                                    p41 = 0;\
                                                                                    p42++; \
                                                                                    if (p42 >= charsetLen) { \
                                                                                      p42 = 0;\
                                                                                      p43++; \
                                                                                      if (p43 >= charsetLen) { \
                                                                                        p43 = 0;\
                                                                                        p44++; \
                                                                                        if (p44 >= charsetLen) { \
                                                                                          p44 = 0;\
                                                                                          p45++; \
                                                                                          if (p45 >= charsetLen) { \
                                                                                            p45 = 0;\
                                                                                            p46++; \
                                                                                            if (p46 >= charsetLen) { \
                                                                                              return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters47Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              p38 = 0;\
                                                                              p39++; \
                                                                              if (p39 >= sharedLengths[39]) { \
                                                                                p39 = 0;\
                                                                                p40++; \
                                                                                if (p40 >= sharedLengths[40]) { \
                                                                                  p40 = 0;\
                                                                                  p41++; \
                                                                                  if (p41 >= sharedLengths[41]) { \
                                                                                    p41 = 0;\
                                                                                    p42++; \
                                                                                    if (p42 >= sharedLengths[42]) { \
                                                                                      p42 = 0;\
                                                                                      p43++; \
                                                                                      if (p43 >= sharedLengths[43]) { \
                                                                                        p43 = 0;\
                                                                                        p44++; \
                                                                                        if (p44 >= sharedLengths[44]) { \
                                                                                          p44 = 0;\
                                                                                          p45++; \
                                                                                          if (p45 >= sharedLengths[45]) { \
                                                                                            p45 = 0;\
                                                                                            p46++; \
                                                                                            if (p46 >= sharedLengths[46]) { \
                                                                                              return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters48() { \
p0++; \
if (p0 >= charsetLen) { \
  p0 = 0;\
  p1++; \
  if (p1 >= charsetLen) { \
    p1 = 0;\
    p2++; \
    if (p2 >= charsetLen) { \
      p2 = 0;\
      p3++; \
      if (p3 >= charsetLen) { \
        p3 = 0;\
        p4++; \
        if (p4 >= charsetLen) { \
          p4 = 0;\
          p5++; \
          if (p5 >= charsetLen) { \
            p5 = 0;\
            p6++; \
            if (p6 >= charsetLen) { \
              p6 = 0;\
              p7++; \
              if (p7 >= charsetLen) { \
                p7 = 0;\
                p8++; \
                if (p8 >= charsetLen) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= charsetLen) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= charsetLen) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= charsetLen) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= charsetLen) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= charsetLen) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= charsetLen) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= charsetLen) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= charsetLen) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= charsetLen) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= charsetLen) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= charsetLen) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= charsetLen) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= charsetLen) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= charsetLen) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= charsetLen) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= charsetLen) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= charsetLen) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= charsetLen) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= charsetLen) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= charsetLen) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= charsetLen) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= charsetLen) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= charsetLen) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= charsetLen) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= charsetLen) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= charsetLen) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= charsetLen) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= charsetLen) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= charsetLen) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= charsetLen) { \
                                                                              p38 = 0;\
                                                                              p39++; \
                                                                              if (p39 >= charsetLen) { \
                                                                                p39 = 0;\
                                                                                p40++; \
                                                                                if (p40 >= charsetLen) { \
                                                                                  p40 = 0;\
                                                                                  p41++; \
                                                                                  if (p41 >= charsetLen) { \
                                                                                    p41 = 0;\
                                                                                    p42++; \
                                                                                    if (p42 >= charsetLen) { \
                                                                                      p42 = 0;\
                                                                                      p43++; \
                                                                                      if (p43 >= charsetLen) { \
                                                                                        p43 = 0;\
                                                                                        p44++; \
                                                                                        if (p44 >= charsetLen) { \
                                                                                          p44 = 0;\
                                                                                          p45++; \
                                                                                          if (p45 >= charsetLen) { \
                                                                                            p45 = 0;\
                                                                                            p46++; \
                                                                                            if (p46 >= charsetLen) { \
                                                                                              p46 = 0;\
                                                                                              p47++; \
                                                                                              if (p47 >= charsetLen) { \
                                                                                                return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

#define incrementCounters48Multi() { \
p0++; \
if (p0 >= sharedLengths[0]) { \
  p0 = 0;\
  p1++; \
  if (p1 >= sharedLengths[1]) { \
    p1 = 0;\
    p2++; \
    if (p2 >= sharedLengths[2]) { \
      p2 = 0;\
      p3++; \
      if (p3 >= sharedLengths[3]) { \
        p3 = 0;\
        p4++; \
        if (p4 >= sharedLengths[4]) { \
          p4 = 0;\
          p5++; \
          if (p5 >= sharedLengths[5]) { \
            p5 = 0;\
            p6++; \
            if (p6 >= sharedLengths[6]) { \
              p6 = 0;\
              p7++; \
              if (p7 >= sharedLengths[7]) { \
                p7 = 0;\
                p8++; \
                if (p8 >= sharedLengths[8]) { \
                  p8 = 0;\
                  p9++; \
                  if (p9 >= sharedLengths[9]) { \
                    p9 = 0;\
                    p10++; \
                    if (p10 >= sharedLengths[10]) { \
                      p10 = 0;\
                      p11++; \
                      if (p11 >= sharedLengths[11]) { \
                        p11 = 0;\
                        p12++; \
                        if (p12 >= sharedLengths[12]) { \
                          p12 = 0;\
                          p13++; \
                          if (p13 >= sharedLengths[13]) { \
                            p13 = 0;\
                            p14++; \
                            if (p14 >= sharedLengths[14]) { \
                              p14 = 0;\
                              p15++; \
                              if (p15 >= sharedLengths[15]) { \
                                p15 = 0;\
                                p16++; \
                                if (p16 >= sharedLengths[16]) { \
                                  p16 = 0;\
                                  p17++; \
                                  if (p17 >= sharedLengths[17]) { \
                                    p17 = 0;\
                                    p18++; \
                                    if (p18 >= sharedLengths[18]) { \
                                      p18 = 0;\
                                      p19++; \
                                      if (p19 >= sharedLengths[19]) { \
                                        p19 = 0;\
                                        p20++; \
                                        if (p20 >= sharedLengths[20]) { \
                                          p20 = 0;\
                                          p21++; \
                                          if (p21 >= sharedLengths[21]) { \
                                            p21 = 0;\
                                            p22++; \
                                            if (p22 >= sharedLengths[22]) { \
                                              p22 = 0;\
                                              p23++; \
                                              if (p23 >= sharedLengths[23]) { \
                                                p23 = 0;\
                                                p24++; \
                                                if (p24 >= sharedLengths[24]) { \
                                                  p24 = 0;\
                                                  p25++; \
                                                  if (p25 >= sharedLengths[25]) { \
                                                    p25 = 0;\
                                                    p26++; \
                                                    if (p26 >= sharedLengths[26]) { \
                                                      p26 = 0;\
                                                      p27++; \
                                                      if (p27 >= sharedLengths[27]) { \
                                                        p27 = 0;\
                                                        p28++; \
                                                        if (p28 >= sharedLengths[28]) { \
                                                          p28 = 0;\
                                                          p29++; \
                                                          if (p29 >= sharedLengths[29]) { \
                                                            p29 = 0;\
                                                            p30++; \
                                                            if (p30 >= sharedLengths[30]) { \
                                                              p30 = 0;\
                                                              p31++; \
                                                              if (p31 >= sharedLengths[31]) { \
                                                                p31 = 0;\
                                                                p32++; \
                                                                if (p32 >= sharedLengths[32]) { \
                                                                  p32 = 0;\
                                                                  p33++; \
                                                                  if (p33 >= sharedLengths[33]) { \
                                                                    p33 = 0;\
                                                                    p34++; \
                                                                    if (p34 >= sharedLengths[34]) { \
                                                                      p34 = 0;\
                                                                      p35++; \
                                                                      if (p35 >= sharedLengths[35]) { \
                                                                        p35 = 0;\
                                                                        p36++; \
                                                                        if (p36 >= sharedLengths[36]) { \
                                                                          p36 = 0;\
                                                                          p37++; \
                                                                          if (p37 >= sharedLengths[37]) { \
                                                                            p37 = 0;\
                                                                            p38++; \
                                                                            if (p38 >= sharedLengths[38]) { \
                                                                              p38 = 0;\
                                                                              p39++; \
                                                                              if (p39 >= sharedLengths[39]) { \
                                                                                p39 = 0;\
                                                                                p40++; \
                                                                                if (p40 >= sharedLengths[40]) { \
                                                                                  p40 = 0;\
                                                                                  p41++; \
                                                                                  if (p41 >= sharedLengths[41]) { \
                                                                                    p41 = 0;\
                                                                                    p42++; \
                                                                                    if (p42 >= sharedLengths[42]) { \
                                                                                      p42 = 0;\
                                                                                      p43++; \
                                                                                      if (p43 >= sharedLengths[43]) { \
                                                                                        p43 = 0;\
                                                                                        p44++; \
                                                                                        if (p44 >= sharedLengths[44]) { \
                                                                                          p44 = 0;\
                                                                                          p45++; \
                                                                                          if (p45 >= sharedLengths[45]) { \
                                                                                            p45 = 0;\
                                                                                            p46++; \
                                                                                            if (p46 >= sharedLengths[46]) { \
                                                                                              p46 = 0;\
                                                                                              p47++; \
                                                                                              if (p47 >= sharedLengths[47]) { \
                                                                                                return;\
}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

