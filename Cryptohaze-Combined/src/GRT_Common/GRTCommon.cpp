/*
Cryptohaze GPU Rainbow Tables
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

#include "GRT_Common/GRTCommon.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#ifdef _WIN32
#include <time.h>
#include <windows.h>
#include <iostream>

using namespace std;

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

// Definition of a gettimeofday function

int gettimeofday(struct timeval *tv, struct timezone *tz)
{
// Define a structure to receive the current Windows filetime
  FILETIME ft;

// Initialize the present time to 0 and the timezone to UTC
  unsigned __int64 tmpres = 0;
  static int tzflag = 0;

  if (NULL != tv)
  {
    GetSystemTimeAsFileTime(&ft);

// The GetSystemTimeAsFileTime returns the number of 100 nanosecond
// intervals since Jan 1, 1601 in a structure. Copy the high bits to
// the 64 bit tmpres, shift it left by 32 then or in the low 32 bits.
    tmpres |= ft.dwHighDateTime;
    tmpres <<= 32;
    tmpres |= ft.dwLowDateTime;

// Convert to microseconds by dividing by 10
    tmpres /= 10;

// The Unix epoch starts on Jan 1 1970.  Need to subtract the difference
// in seconds from Jan 1 1601.
    tmpres -= DELTA_EPOCH_IN_MICROSECS;

// Finally change microseconds to seconds and place in the seconds value.
// The modulus picks up the microseconds.
    tv->tv_sec = (long)(tmpres / 1000000UL);
    tv->tv_usec = (long)(tmpres % 1000000UL);
  }

  if (NULL != tz)
  {
    if (!tzflag)
    {
      _tzset();
      tzflag++;
    }

// Adjust for the timezone west of Greenwich
      tz->tz_minuteswest = _timezone / 60;
    tz->tz_dsttime = _daylight;
  }

  return 0;
}

#endif




// Return true if d1 is less than d2
bool tableDataSortPredicate(const hashPasswordData &d1, const hashPasswordData &d2) {
    int i;
    for (i = 0; i < MAX_HASH_LENGTH_BYTES; i++) {
        if (d1.hash[i] == d2.hash[i]) {
            continue;
        } else if (d1.hash[i] > d2.hash[i]) {
            return 0;
        } else if (d1.hash[i] < d2.hash[i]) {
            return 1;
        }
    }
    // Exactly equal = return 0.
    return 0;
}

// Return true if d1 is less than d2
bool hashDataSortPredicate(const hashData &d1, const hashData &d2) {
    int i;
    for (i = 0; i < MAX_HASH_LENGTH_BYTES; i++) {
        if (d1.hash[i] > d2.hash[i]) {
            return 0;
        } else if (d1.hash[i] < d2.hash[i]) {
            return 1;
        }
    }
    // Exactly equal = return 0.
    return 0;
}

// Return true if equal, else return false.
bool hashDataUniquePredicate(const hashData &d1, const hashData &d2) {
    int i;
    for (i = 0; i < MAX_HASH_LENGTH_BYTES; i++) {
        if (d1.hash[i] != d2.hash[i]) {
            return 0;
        }
    }
    // Exactly equal = return 1.
    return 1;
}


bool passwordDataSortPredicate(const hashPasswordData &d1, const hashPasswordData &d2) {
    int i;
    for (i = 0; i < MAX_PASSWORD_LENGTH; i++) {
        if (d1.password[i] > d2.password[i]) {
            return 0;
        } else if (d1.password[i] < d2.password[i]) {
            return 1;
        }
    }
    // Exactly equal = return 0.
    return 0;

}

bool passwordDataUniquePredicate(const hashPasswordData &d1, const hashPasswordData &d2) {
    int i;
    for (i = 0; i < MAX_PASSWORD_LENGTH; i++) {
        if (d1.password[i] != d2.password[i]) {
            return 0;
        }
    }
    // Exactly equal = return 1.
    return 1;
}



/* convertAsciiToBinary takes an ASCII hex string *input, and converts it to "true binary" in *hash.
 * This will only convert up to maxLength bytes or the end of input.  It drops everything beyond that.
 * Returns number of characters converted.
 * Behavior with non-hex input characters is undefined.
 */
int convertAsciiToBinary(const char *input, unsigned char *hash, int maxLength) {
  char convertSpace[10];
  int i;

  // Loop until either maxLength is hit, or strlen(intput) / 2 is hit.
  for (i = 0; (i < maxLength) && (i < (strlen(input) / 2)); i++) {
    convertSpace[0] = input[2 * i];
    convertSpace[1] = input[2 * i + 1];
    convertSpace[2] = 0;
    sscanf(convertSpace, "%x", &hash[i]);
  }
  return i;
}




char getTableVersion(const char *filename){

    static const char MAGIC_0 = 'G';
    static const char MAGIC_1 = 'R';
    static const char MAGIC_2 = 'T';

    FILE *Table;

    // Read the first 4 bytes of table header.
    char Table_Header[4];

    // Open as a large file
    Table = fopen(filename, "rb");

    // If the table file can't be opened, return false.
    if (Table == NULL) {
        printf("Cannot open table %s: fopen failed.\n", filename);
        return -1;
    }

    memset(Table_Header, 0, 4);

    // If the read fails, clean up and return false.
    if (!fread(Table_Header, 4, 1, Table)) {
        fclose(Table);
        return -1;
    }

    fclose(Table);

    // Some sanity checks
    if ((Table_Header[0] != MAGIC_0) || (Table_Header[1] != MAGIC_1) ||
            (Table_Header[2] != MAGIC_2)) {
        printf("Table magic does not match!\n");
        return -1;
    }

    // Return the table version
    return Table_Header[3];
}

