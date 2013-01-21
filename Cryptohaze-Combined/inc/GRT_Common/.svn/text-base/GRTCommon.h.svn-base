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

#ifndef _GRTCOMMON_H
#define	_GRTCOMMON_H

#ifdef _WIN32
#include "windows/stdint.h"
#else
#include <stdint.h>
#endif

typedef uint32_t UINT4;

#define GPU_CHARSET_SIZE 512
#define MAX_HASH_LENGTH_BYTES 32
#define MAX_PASSWORD_LENGTH 32
#define MAX_CHARSET_LENGTH 256

#define DEFAULT_CUDA_EXECUTION_TIME 50
#define DEFAULT_CUDA_EXECUTION_TIME_NO_TIMEOUT 500


// Maximum supported GPU/CPU thread count.  Updating this will require
// updating the display code to scroll the rates.
#define MAX_SUPPORTED_THREADS 16


// USE_BOOST_MMAP: Use boost memory mapped files if set, else use posix mmap.
// If USE_BOOST_THREADS is 1, boost::thread is used.  Else, pthreads are used.

#ifdef _WIN32
#define USE_BOOST_MMAP 1
#define USE_BOOST_THREADS 1
#else
#define USE_BOOST_MMAP 0
#define USE_BOOST_THREADS 0
#endif


#if !USE_BOOST_THREADS
#include <pthread.h>
#endif

#ifdef _WIN32
struct timezone
{
  int  tz_minuteswest; /* minutes W of Greenwich */
  int  tz_dsttime;     /* type of dst correction */
};

// this is here to bring in struct timeval
#include <WinSock2.h>

int gettimeofday(struct timeval *tv, struct timezone *tz);
#endif

typedef struct hashPasswordData {
  unsigned char hash[MAX_HASH_LENGTH_BYTES];
  unsigned char password[MAX_PASSWORD_LENGTH];
} hashPasswordData;


typedef struct hashData {
  unsigned char hash[MAX_HASH_LENGTH_BYTES];
} hashData;

// Oh GFF, structure packing.
#pragma pack(push)
#pragma pack(1)
// The index file structure
typedef struct indexFile {
    uint32_t Index;
    uint64_t Offset;
} indexFile;
#pragma pack(pop)

// This is a structure to pass global commands to ALL threads.
typedef struct global_commands {
    char pause;
    char exit;
    int  exit_after_count; // Exit after this many hashes have been found.
    char exit_message[256]; // Exit message, if set.
} global_commands;


unsigned char lessThanEqual(hashPasswordData *a, hashPasswordData *b, int hashLength = 16);
unsigned char greaterThanEqual(hashPasswordData *a, hashPasswordData *b, int hashLength = 16);

bool tableDataSortPredicate(const hashPasswordData &d1, const hashPasswordData &d2);
bool hashDataSortPredicate(const hashData &d1, const hashData &d2);
bool hashDataUniquePredicate(const hashData &d1, const hashData &d2);

bool passwordDataSortPredicate(const hashPasswordData &d1, const hashPasswordData &d2);
bool passwordDataUniquePredicate(const hashPasswordData &d1, const hashPasswordData &d2);




int convertAsciiToBinary(const char *input, unsigned char *hash, int maxLength);


char getTableVersion(const char *filename);

void chomp(char *s);

// mtwist.h defines to make me sane
/*
extern "C" void	mt_goodseed(void);
extern "C" unsigned long long mt_llrand();
extern "C" void	mt_seed32new(unsigned long seed);
extern "C" unsigned long mt_lrand(void);
*/
/*
extern void mt_goodseed(void);
extern unsigned long long mt_llrand();
extern void mt_seed32new(unsigned long seed);
extern unsigned long mt_lrand(void);
*/

#endif	/* _GRTCOMMON_H */
