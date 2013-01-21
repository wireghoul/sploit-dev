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

#ifndef CH_COMMON_H
#define CH_COMMON_H

// Set to 1 to use Boost threading on Windows.
// Else use pthreads.  Bitweasil likes pthreads.
#ifdef _WIN32
	#define USE_BOOST_THREADS 1
#else
	#define USE_BOOST_THREADS 0
#endif

// Set to 1 if network support is desired
#define USE_NETWORK 1

#ifdef _WIN32
#define _WIN32_WINNT 0x501
#endif

// Include all the host-only stuff
#ifndef __CUDACC__
	#ifdef _WIN32
        #ifndef _WINSOCKAPI_
    		#define _WINSOCKAPI_ // Prevent inclusion of winsock.h in windows.h
        #endif
        #ifndef WIN32_LEAN_AND_MEAN
		    #define WIN32_LEAN_AND_MEAN
        #endif

		#include <winsock2.h>
		#include <curses.h>
		#include <argtable2.h>
		#include <time.h>
		#include <windows.h>
		#include <iostream>
	#else
		#include <argtable2.h>
		#include <unistd.h>
		#include <ncurses.h>
                #include <sys/time.h>
	#endif

	#if USE_BOOST_THREADS
		#include <boost/thread.hpp>
		#include <boost/thread/mutex.hpp>
	#else
		#include <pthread.h>
	#endif

	#if USE_NETWORK
		#include <boost/array.hpp>
		#include <boost/asio.hpp>
		#include <boost/bind.hpp>
		#include <boost/thread.hpp>
	#endif

	// C++ Headers
	#include <iostream>
	#include <valarray>
	#include <new>
	#include <exception>
#endif

// Other includes.
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <CUDA_Common/CUDA_SAFE_CALL.h>

// C headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <math.h>
#include <signal.h>
#include <limits.h>

#ifdef _WIN32
	#include "windows/stdint.h"
#else
	#include <stdint.h>
#endif

#ifdef _WIN32
	#define CHSleep(x) Sleep(x*1000)
#else
	#define CHSleep(x) sleep(x)
#endif

// Custom stuff
#include "CH_Common/b64.h"







#define MAX_PASSWORD_LEN 48
#define MAX_PASSWORD_LEN_16 16
#define MAX_CHARSET_LENGTH 128

#define DEFAULT_CUDA_THREADS 512
#define DEFAULT_CUDA_BLOCKS 120
#define DEFAULT_CUDA_EXECUTION_TIME 100
#define DEFAULT_CUDA_EXECUTION_TIME_NO_TIMEOUT 500

// :) <3
#define DEFAULT_NETWORK_PORT 12410
#define NETWORK_WAIT_TIME 5


// This uses 4kb of constant memory, which we can do
#define MAX_MSSQL_HASHES 1024

#define MAX_SALTED_HASHES 1024


#define MD5_HASH_LENGTH 16
#define MD4_HASH_LENGTH 16
#define NTLM_HASH_LENGTH 16
#define SHA1_HASH_LENGTH 20

#define MAX_SALT_LENGTH 16

#define SALT_IS_FIRST 1
#define SALT_IS_LAST 0

#define SALT_IS_LITERAL 1
#define SALT_IS_HEX 0



// Some general defines for hashes
#define MAX_HASH_TYPES 100
#define MAX_HASH_STRING_LENGTH 32
#define MAX_HASH_DESCRIPTION_LENGTH 4096
#define MAX_HASH_ALGORITHM_LENGTH 256

typedef struct start_positions {
  unsigned char p0;
  unsigned char p1;
  unsigned char p2;
  unsigned char p3;
  unsigned char p4;
  unsigned char p5;
  unsigned char p6;
  unsigned char p7;
  unsigned char p8;
  unsigned char p9;
  unsigned char p10;
  unsigned char p11;
  unsigned char p12;
  unsigned char p13;
  unsigned char p14;
  unsigned char p15;
  unsigned char p16;
  unsigned char p17;
  unsigned char p18;
  unsigned char p19;
  unsigned char p20;
  unsigned char p21;
  unsigned char p22;
  unsigned char p23;
  unsigned char p24;
  unsigned char p25;
  unsigned char p26;
  unsigned char p27;
  unsigned char p28;
  unsigned char p29;
  unsigned char p30;
  unsigned char p31;
  unsigned char p32;
  unsigned char p33;
  unsigned char p34;
  unsigned char p35;
  unsigned char p36;
  unsigned char p37;
  unsigned char p38;
  unsigned char p39;
  unsigned char p40;
  unsigned char p41;
  unsigned char p42;
  unsigned char p43;
  unsigned char p44;
  unsigned char p45;
  unsigned char p46;
  unsigned char p47;
}start_positions;

// This is a structure to pass global commands to ALL threads.
typedef struct global_commands {
    char pause; // Pause work
    char exit;  // Exit current task
    char user_exit; // Exit caused by a user (ctrl-c, q)
    int  exit_after_count; // Exit after this many hashes have been found.
    char exit_message[256]; // Exit message, if set.
} global_commands;

long int file_size(const char *path);

int convertAsciiToBinary(char *input, unsigned char *hash, int maxLength);

void printCudaDeviceInfo(int deviceId);

// Returns the number of stream processors a device has for basic auto-tune.
int getCudaStreamProcessorCount(int deviceId);

// Returns true if the device has a timeout set.
int getCudaHasTimeout(int deviceId);

// Get the default thread & block counts
int getCudaDefaultThreadCountBySPCount(int streamProcCount);
int getCudaDefaultBlockCountBySPCount(int streamProcCount);

// Returns true if compute capability is >= 2
int getCudaIsFermi(int deviceId);


inline int ConvertSMVer2Cores(int major, int minor);

void chomp(char *s);

#ifdef _WIN32
struct timezone
{
  int  tz_minuteswest; /* minutes W of Greenwich */
  int  tz_dsttime;     /* type of dst correction */
};

int gettimeofday(struct timeval *tv, struct timezone *tz);
#endif


#endif
