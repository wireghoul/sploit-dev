/**
 * This file contains various defines that are used throughout the code.
 * Ideally, changing stuff here will change it everywhere with a recompile.
 * 
 * Keep this file clean - it will be included in CUDA kernels and OpenCL kernels.
 * 
 * This means no fancy expansions, no includes, no classes.  JUST DEFINES.
 */

#ifndef __MFNDEFINES_H_
#define __MFNDEFINES_H_

/**
 * The maximum password length supported by the plan hashfiles.
 */
#define MFN_HASH_TYPE_PLAIN_MAX_PASSLEN 48

/**
 * The maximum password length supported by wordlist stuff.
 */
#define MFN_HASH_TYPE_WORDLIST_MAX_PASSLEN 128

/**
 * The maximum charset length supported by the plain hashfiles.
 */
#define MFN_HASH_TYPE_PLAIN_MAX_CHARSET_LENGTH 128

/**
 * Display class defines
 */
#define UNUSED_THREAD 0
#define GPU_THREAD 1
#define CPU_THREAD 2
#define NETWORK_HOST 3

#define SYSTEM_MODE_STANDALONE 1
#define SYSTEM_MODE_SERVER 2
#define SYSTEM_MODE_CLIENT 3


//=============== Defines for the hash types ====================
// Default unspecified hash type.
#define MFN_HASHTYPE_UNDEFINED 0x000

// Plain hashes: 0x1000 prefix
  // MD5: 0x000 prefix
    // MD5 Plain: Unsalted, arbitrary numbers
    #define MFN_HASHTYPE_PLAIN_MD5                  0x1000
    // MD5 Single: Unsalted, one hash.
    #define MFN_HASHTYPE_PLAIN_MD5_SINGLE           0x1001

  // MD5 Double: 0x0100 prefix
  // md5(md5($pass))
  #define MFN_HASHTYPE_DOUBLE_MD5                   0x1100

  // MD5 Triple: 0x0200 prefix
  // md5(md5(md5($pass)))
  #define MFN_HASHTYPE_TRIPLE_MD5                   0x1200
    
  // MD5 Duplicated: 0x0300 prefix
  // md5($pass.$pass)
  #define MFN_HASHTYPE_DUPLICATED_MD5               0x1300


  // NTLM: 0x400 prefix
  #define MFN_HASHTYPE_NTLM                         0x1400
  #define MFN_HASHTYPE_NTLM_SINGLE                  0x1401

  // NTLM Duplicated: 0x500 prefix
  #define MFN_HASHTYPE_DUPLICATED_NTLM              0x1500

  // LM Hashes: 0x600 prefix
  #define MFN_HASHTYPE_LM                           0x1600
  
  // SHA1: 0x700
  #define MFN_HASHTYPE_SHA1                         0x1700

  #define MFN_HASHTYPE_SHA256                       0x1800
  #define MFN_HASHTYPE_DOUBLE_SHA256                0x1801

  #define MFN_HASHTYPE_PLAIN_LOTUS                  0x1900

// Salted: 0x2000 prefix
#define MFN_HASHTYPE_MD5_PS                         0x2000
#define MFN_HASHTYPE_IPB                            0x2001
#define MFN_HASHTYPE_PHPASS                         0x2002

// Multiple algorithm kernels
#define MFN_HASHTYPE_16HEX                          0x3000
#define MFN_HASHTYPE_20HEX                          0x3001
#define MFN_HASHTYPE_32HEX                          0x3002

// Wordlist kernels
#define MFN_HASHTYPE_PLAIN_MD5WL                    0x4000
#define MFN_HASHTYPE_PLAIN_NTLMWL                   0x4001
#define MFN_HASHTYPE_IPBWL                          0x4002
#define MFN_HASHTYPE_PHPASSWL                       0x4003


//============ Defines for the hash factory ===========


// CHCharsetNew class identifier
#define CH_CHARSET_NEW_CLASS_ID 0x1000

// MFNWorkunit class identifiers
#define MFN_WORKUNIT_ROBUST_CLASS_ID 0x2000
#define MFN_WORKUNIT_NETWORK_CLASS_ID 0x2001
#define MFN_WORKUNIT_WORDLIST_CLASS_ID 0x2002

// CHDisplay class identifiers
#define MFN_DISPLAY_CLASS_CURSES 0x3000
#define MFN_DISPLAY_CLASS_DEBUG 0x3001
#define MFN_DISPLAY_CLASS_DAEMON 0x3002
#define MFN_DISPLAY_CLASS_GUI 0x3003

// CHHashfile identifiers
#define CH_HASHFILE_PLAIN_16 0x4000
#define CH_HASHFILE_PLAIN_20 0x4001
#define CH_HASHFILE_PLAIN_32 0x4002
#define CH_HASHFILE_LM       0x4100
#define CH_HASHFILE_SALTED_32_PASS_SALT  0x4200
#define CH_HASHFILE_IPB  0x4201
#define CH_HASHFILE_PHPASS  0x4300

// Commandlinedata identifiers
#define CH_COMMANDLINEDATA 0x6000
#define CH_COMMANDLINEDATAGUI 0x6001

// Wordlist class identifier
#define CH_WORDLISTCLASS 0x7000

// CHCUDAUtils identifiers
#define CH_CUDAUTILS 0x7000

// MFNHashIdentifiers identifiers
#define MFN_HASHIDENTIFIERS 0x8000

// Hashfile identifiers
#define CH_HASHTYPE_PLAIN_CUDA_MD5 0x10001

#define CH_HASHTYPE_PLAIN_OPENCL_MD5 0x20001

/**
 * Value for the default/invalid class ID.  This is set for classes that do not
 * have a default type.
 */
#define CH_CHARSET_CLASS_INVALID_ID 0xFFFFFFFF


// Byte-length identifiers for various password algorithms.
// These are used by the GPU for reporting what it found in multi-algorithm kernels.
#define MFN_PASSWORD_NOT_FOUND   0x00
#define MFN_PASSWORD_SINGLE_MD5  0x01
#define MFN_PASSWORD_DOUBLE_MD5  0x02
#define MFN_PASSWORD_TRIPLE_MD5  0x03
#define MFN_PASSWORD_NTLM        0x04
#define MFN_PASSWORD_SHA1        0x05
#define MFN_PASSWORD_SHA1_OF_MD5 0x06
#define MFN_PASSWORD_MD5_OF_SHA1 0x07
#define MFN_PASSWORD_LM          0x08
#define MFN_PASSWORD_SHA256      0x09
#define MFN_PASSWORD_MD4         0x0A
#define MFN_PASSWORD_LOTUS       0x0B

// Windows vs Linux defines for sleep - no issue in GPU kernels.
#ifdef _WIN32
	#define CHSleep(x) Sleep(x*1000)
#else
	#define CHSleep(x) sleep(x)
#endif

#endif