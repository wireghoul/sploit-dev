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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static const char  table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static const int   BASE64_INPUT_SIZE = 57;

char isbase64(char c)
{
   return c && strchr(table, c) != NULL;
}

inline char value(char c)
{
   const char *p = strchr(table, c);
   if(p) {
      return p-table;
   } else {
      return 0;
   }
}

int UnBase64(unsigned char *dest, const unsigned char *src, int srclen)
{
   *dest = 0;
   if(*src == 0)
   {
      return 0;
   }
   unsigned char *p = dest;
   do
   {

      char a = value(src[0]);
      char b = value(src[1]);
      char c = value(src[2]);
      char d = value(src[3]);
      *p++ = (a << 2) | (b >> 4);
      *p++ = (b << 4) | (c >> 2);
      *p++ = (c << 6) | d;
      if(!isbase64(src[1]))
      {
         p -= 2;
         break;
      }
      else if(!isbase64(src[2]))
      {
         p -= 2;
         break;
      }
      else if(!isbase64(src[3]))
      {
         p--;
         break;
      }
      src += 4;
      while(*src && (*src == 13 || *src == 10)) src++;
      printf("srclen: %d\n", srclen);
   } while(srclen-= 4);
   *p = 0;
   return p-dest;
}



#include <stdio.h>

unsigned char alphabet[65] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int
Base64(unsigned char *src, unsigned char *dest, int srclen)
{
    long int cols, bits, c, char_count, output_pos;

    output_pos = 0;
    char_count = 0;
    bits = 0;
    cols = 0;
    int i;
    for (i = 0; i < srclen; i++) {
        c = src[i];
	bits += c;
	char_count++;
	if (char_count == 3) {
	    dest[output_pos++] = (alphabet[bits >> 18]);
	    dest[output_pos++] = (alphabet[(bits >> 12) & 0x3f]);
	    dest[output_pos++] = (alphabet[(bits >> 6) & 0x3f]);
	    dest[output_pos++] = (alphabet[bits & 0x3f]);
	    bits = 0;
	    char_count = 0;
	} else {
	    bits <<= 8;
	}
    }
    if (char_count != 0) {
	bits <<= 16 - (8 * char_count);
	dest[output_pos++] = (alphabet[bits >> 18]);
	dest[output_pos++] = (alphabet[(bits >> 12) & 0x3f]);
	if (char_count == 1) {
	    dest[output_pos++] = ('=');
	    dest[output_pos++] = ('=');
	} else {
	    dest[output_pos++] = (alphabet[(bits >> 6) & 0x3f]);
	    dest[output_pos++] = ('=');
	}

    }
    return 1;
}



