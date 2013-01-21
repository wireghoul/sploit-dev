/*
Cryptohaze Multiforcer & Wordyforcer - low performance GPU password cracking
Copyright (C) 2012  Bitweasil (http://www.cryptohaze.com/)

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

/**
 * This file contains functions related to searching the global hashlist and
 * copying passwords into the host space when found.
 */


/**
 * Little Endian Wordlist Functions - this will copy from a little endian
 * wordlist to the found password region for output.  This is for 128 bit hash
 * types.
 */

// dfp: Device Found Passwords
// dfpf: Device Found Passwords Flags
#ifdef grt_vector_1
#define CopyFoundPasswordToMemoryFromWordlistLE(dfp, dfpf, suffix) { \
    vector_type passwordLength = convert_type(vload_type(get_global_id(0), \
        &deviceWordlistLengths[passwordStep * deviceNumberThreads])); \
    vector_type wordlistBlock; \
    /*printf("Got password to copy length %d\n", passwordLength);*/ \
    for (uint i = 0; i < passwordLength; i++) { \
        if ((i % 4) == 0) { \
            /*printf("Loading block %d\n", i / 4);*/ \
            wordlistBlock = vload_type(get_global_id(0), \
                &deviceWordlistBlocks[(i / 4) * deviceNumberWords + \
                passwordStep * deviceNumberThreads]); \
        } \
        dfp[search_index * MFN_HASH_TYPE_WORDLIST_MAX_PASSLEN + i] = (wordlistBlock >> ((i % 4) * 8)) & 0xff; \
    } \
    dfpf[search_index] = (unsigned char) 1; \
}
#else
#define CopyFoundPasswordToMemoryFromWordlistLE(dfp, dfpf, suffix) { \
    vector_type passwordLength = convert_type(vload_type(get_global_id(0), \
        &deviceWordlistLengths[passwordStep * deviceNumberThreads])); \
    vector_type wordlistBlock; \
    /*printf("Got password to copy length %d\n", passwordLength.s##suffix);*/ \
    for (uint i = 0; i < passwordLength.s##suffix; i++) { \
        if ((i % 4) == 0) { \
            /*printf("Loading block %d\n", i / 4);*/ \
            wordlistBlock = vload_type(get_global_id(0), \
                &deviceWordlistBlocks[(i / 4) * deviceNumberWords + \
                passwordStep * deviceNumberThreads]); \
        } \
        dfp[search_index * MFN_HASH_TYPE_WORDLIST_MAX_PASSLEN + i] = (wordlistBlock.s##suffix >> ((i % 4) * 8)) & 0xff; \
    } \
    dfpf[search_index] = (unsigned char) 1; \
}
#endif


#ifdef grt_vector_1
#define CheckWordlistPassword128LE(dgh, dfp, dfpf, dnh, suffix) { \
    search_high = dnh; \
    search_low = 0; \
    while (search_low < search_high) { \
        search_index = search_low + (search_high - search_low) / 2; \
        current_hash_value = dgh[4 * search_index]; \
        if (current_hash_value < a) { \
            search_low = search_index + 1; \
        } else { \
            search_high = search_index; \
        } \
        if ((a == current_hash_value) && (search_low < dnh)) { \
            break; \
        } \
    } \
    if (a == current_hash_value) { \
        while (search_index && (a == dgh[(search_index - 1) * 4])) { \
            search_index--; \
        } \
        while ((a == dgh[search_index * 4])) { \
            if (b == dgh[search_index * 4 + 1]) { \
                if (c == dgh[search_index * 4 + 2]) { \
                    if (d == dgh[search_index * 4 + 3]) { \
                    CopyFoundPasswordToMemoryFromWordlistLE(dfp, dfpf, suffix); \
                    } \
                } \
            } \
            search_index++; \
        } \
    } \
}
#else
#define CheckWordlistPassword128LE(dgh, dfp, dfpf, dnh, suffix) { \
    search_high = dnh; \
    search_low = 0; \
    while (search_low < search_high) { \
        search_index = search_low + (search_high - search_low) / 2; \
        current_hash_value = dgh[4 * search_index]; \
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
        while (search_index && (a.s##suffix == dgh[(search_index - 1) * 4])) { \
            search_index--; \
        } \
        while ((a.s##suffix == dgh[search_index * 4])) { \
            if (b.s##suffix == dgh[search_index * 4 + 1]) { \
                if (c.s##suffix == dgh[search_index * 4 + 2]) { \
                    if (d.s##suffix == dgh[search_index * 4 + 3]) { \
                    CopyFoundPasswordToMemoryFromWordlistLE(dfp, dfpf, suffix); \
                    } \
                } \
            } \
            search_index++; \
        } \
    } \
}
#endif

/**
 * Little endian copy from b0-b13, for lengths 0-55, MD5-style data with all
 * the bytes in each word filled with a character (as opposed to NTLM).
 */


// dfp: Device Found Passwords
// dfpf: Device Found Passwords Flags
#ifdef grt_vector_1
#define CopyFoundPasswordToMemoryLE(dfp, dfpf, suffix) { \
    switch ( PASSWORD_LENGTH ) { \
        case 55: \
            dfp[search_index * PASSWORD_LENGTH + 54] = (b13 >> 16) & 0xff; \
        case 54: \
            dfp[search_index * PASSWORD_LENGTH + 53] = (b13 >> 8) & 0xff; \
        case 53: \
            dfp[search_index * PASSWORD_LENGTH + 52] = (b13 >> 0) & 0xff; \
        case 52: \
            dfp[search_index * PASSWORD_LENGTH + 51] = (b12 >> 24) & 0xff; \
        case 51: \
            dfp[search_index * PASSWORD_LENGTH + 50] = (b12 >> 16) & 0xff; \
        case 50: \
            dfp[search_index * PASSWORD_LENGTH + 49] = (b12 >> 8) & 0xff; \
        case 49: \
            dfp[search_index * PASSWORD_LENGTH + 48] = (b12 >> 0) & 0xff; \
        case 48: \
            dfp[search_index * PASSWORD_LENGTH + 47] = (b11 >> 24) & 0xff; \
        case 47: \
            dfp[search_index * PASSWORD_LENGTH + 46] = (b11 >> 16) & 0xff; \
        case 46: \
            dfp[search_index * PASSWORD_LENGTH + 45] = (b11 >> 8) & 0xff; \
        case 45: \
            dfp[search_index * PASSWORD_LENGTH + 44] = (b11 >> 0) & 0xff; \
        case 44: \
            dfp[search_index * PASSWORD_LENGTH + 43] = (b10 >> 24) & 0xff; \
        case 43: \
            dfp[search_index * PASSWORD_LENGTH + 42] = (b10 >> 16) & 0xff; \
        case 42: \
            dfp[search_index * PASSWORD_LENGTH + 41] = (b10 >> 8) & 0xff; \
        case 41: \
            dfp[search_index * PASSWORD_LENGTH + 40] = (b10 >> 0) & 0xff; \
        case 40: \
            dfp[search_index * PASSWORD_LENGTH + 39] = (b9 >> 24) & 0xff; \
        case 39: \
            dfp[search_index * PASSWORD_LENGTH + 38] = (b9 >> 16) & 0xff; \
        case 38: \
            dfp[search_index * PASSWORD_LENGTH + 37] = (b9 >> 8) & 0xff; \
        case 37: \
            dfp[search_index * PASSWORD_LENGTH + 36] = (b9 >> 0) & 0xff; \
        case 36: \
            dfp[search_index * PASSWORD_LENGTH + 35] = (b8 >> 24) & 0xff; \
        case 35: \
            dfp[search_index * PASSWORD_LENGTH + 34] = (b8 >> 16) & 0xff; \
        case 34: \
            dfp[search_index * PASSWORD_LENGTH + 33] = (b8 >> 8) & 0xff; \
        case 33: \
            dfp[search_index * PASSWORD_LENGTH + 32] = (b8 >> 0) & 0xff; \
        case 32: \
            dfp[search_index * PASSWORD_LENGTH + 31] = (b7 >> 24) & 0xff; \
        case 31: \
            dfp[search_index * PASSWORD_LENGTH + 30] = (b7 >> 16) & 0xff; \
        case 30: \
            dfp[search_index * PASSWORD_LENGTH + 29] = (b7 >> 8) & 0xff; \
        case 29: \
            dfp[search_index * PASSWORD_LENGTH + 28] = (b7 >> 0) & 0xff; \
        case 28: \
            dfp[search_index * PASSWORD_LENGTH + 27] = (b6 >> 24) & 0xff; \
        case 27: \
            dfp[search_index * PASSWORD_LENGTH + 26] = (b6 >> 16) & 0xff; \
        case 26: \
            dfp[search_index * PASSWORD_LENGTH + 25] = (b6 >> 8) & 0xff; \
        case 25: \
            dfp[search_index * PASSWORD_LENGTH + 24] = (b6 >> 0) & 0xff; \
        case 24: \
            dfp[search_index * PASSWORD_LENGTH + 23] = (b5 >> 24) & 0xff; \
        case 23: \
            dfp[search_index * PASSWORD_LENGTH + 22] = (b5 >> 16) & 0xff; \
        case 22: \
            dfp[search_index * PASSWORD_LENGTH + 21] = (b5 >> 8) & 0xff; \
        case 21: \
            dfp[search_index * PASSWORD_LENGTH + 20] = (b5 >> 0) & 0xff; \
        case 20: \
            dfp[search_index * PASSWORD_LENGTH + 19] = (b4 >> 24) & 0xff; \
        case 19: \
            dfp[search_index * PASSWORD_LENGTH + 18] = (b4 >> 16) & 0xff; \
        case 18: \
            dfp[search_index * PASSWORD_LENGTH + 17] = (b4 >> 8) & 0xff; \
        case 17: \
            dfp[search_index * PASSWORD_LENGTH + 16] = (b4 >> 0) & 0xff; \
        case 16: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b3 >> 24) & 0xff; \
        case 15: \
            dfp[search_index * PASSWORD_LENGTH + 14] = (b3 >> 16) & 0xff; \
        case 14: \
            dfp[search_index * PASSWORD_LENGTH + 13] = (b3 >> 8) & 0xff; \
        case 13: \
            dfp[search_index * PASSWORD_LENGTH + 12] = (b3 >> 0) & 0xff; \
        case 12: \
            dfp[search_index * PASSWORD_LENGTH + 11] = (b2 >> 24) & 0xff; \
        case 11: \
            dfp[search_index * PASSWORD_LENGTH + 10] = (b2 >> 16) & 0xff; \
        case 10: \
            dfp[search_index * PASSWORD_LENGTH + 9] = (b2 >> 8) & 0xff; \
        case 9: \
            dfp[search_index * PASSWORD_LENGTH + 8] = (b2 >> 0) & 0xff; \
        case 8: \
            dfp[search_index * PASSWORD_LENGTH + 7] = (b1 >> 24) & 0xff; \
        case 7: \
            dfp[search_index * PASSWORD_LENGTH + 6] = (b1 >> 16) & 0xff; \
        case 6: \
            dfp[search_index * PASSWORD_LENGTH + 5] = (b1 >> 8) & 0xff; \
        case 5: \
            dfp[search_index * PASSWORD_LENGTH + 4] = (b1 >> 0) & 0xff; \
        case 4: \
            dfp[search_index * PASSWORD_LENGTH + 3] = (b0 >> 24) & 0xff; \
        case 3: \
            dfp[search_index * PASSWORD_LENGTH + 2] = (b0 >> 16) & 0xff; \
        case 2: \
            dfp[search_index * PASSWORD_LENGTH + 1] = (b0 >> 8) & 0xff; \
        case 1: \
            dfp[search_index * PASSWORD_LENGTH + 0] = (b0 >> 0) & 0xff; \
    } \
    dfpf[search_index] = (unsigned char) 1; \
}
#else
#define CopyFoundPasswordToMemoryLE(dfp, dfpf, suffix) { \
    switch ( PASSWORD_LENGTH ) { \
        case 55: \
            dfp[search_index * PASSWORD_LENGTH + 54] = (b13.s##suffix >> 16) & 0xff; \
        case 54: \
            dfp[search_index * PASSWORD_LENGTH + 53] = (b13.s##suffix >> 8) & 0xff; \
        case 53: \
            dfp[search_index * PASSWORD_LENGTH + 52] = (b13.s##suffix >> 0) & 0xff; \
        case 52: \
            dfp[search_index * PASSWORD_LENGTH + 51] = (b12.s##suffix >> 24) & 0xff; \
        case 51: \
            dfp[search_index * PASSWORD_LENGTH + 50] = (b12.s##suffix >> 16) & 0xff; \
        case 50: \
            dfp[search_index * PASSWORD_LENGTH + 49] = (b12.s##suffix >> 8) & 0xff; \
        case 49: \
            dfp[search_index * PASSWORD_LENGTH + 48] = (b12.s##suffix >> 0) & 0xff; \
        case 48: \
            dfp[search_index * PASSWORD_LENGTH + 47] = (b11.s##suffix >> 24) & 0xff; \
        case 47: \
            dfp[search_index * PASSWORD_LENGTH + 46] = (b11.s##suffix >> 16) & 0xff; \
        case 46: \
            dfp[search_index * PASSWORD_LENGTH + 45] = (b11.s##suffix >> 8) & 0xff; \
        case 45: \
            dfp[search_index * PASSWORD_LENGTH + 44] = (b11.s##suffix >> 0) & 0xff; \
        case 44: \
            dfp[search_index * PASSWORD_LENGTH + 43] = (b10.s##suffix >> 24) & 0xff; \
        case 43: \
            dfp[search_index * PASSWORD_LENGTH + 42] = (b10.s##suffix >> 16) & 0xff; \
        case 42: \
            dfp[search_index * PASSWORD_LENGTH + 41] = (b10.s##suffix >> 8) & 0xff; \
        case 41: \
            dfp[search_index * PASSWORD_LENGTH + 40] = (b10.s##suffix >> 0) & 0xff; \
        case 40: \
            dfp[search_index * PASSWORD_LENGTH + 39] = (b9.s##suffix >> 24) & 0xff; \
        case 39: \
            dfp[search_index * PASSWORD_LENGTH + 38] = (b9.s##suffix >> 16) & 0xff; \
        case 38: \
            dfp[search_index * PASSWORD_LENGTH + 37] = (b9.s##suffix >> 8) & 0xff; \
        case 37: \
            dfp[search_index * PASSWORD_LENGTH + 36] = (b9.s##suffix >> 0) & 0xff; \
        case 36: \
            dfp[search_index * PASSWORD_LENGTH + 35] = (b8.s##suffix >> 24) & 0xff; \
        case 35: \
            dfp[search_index * PASSWORD_LENGTH + 34] = (b8.s##suffix >> 16) & 0xff; \
        case 34: \
            dfp[search_index * PASSWORD_LENGTH + 33] = (b8.s##suffix >> 8) & 0xff; \
        case 33: \
            dfp[search_index * PASSWORD_LENGTH + 32] = (b8.s##suffix >> 0) & 0xff; \
        case 32: \
            dfp[search_index * PASSWORD_LENGTH + 31] = (b7.s##suffix >> 24) & 0xff; \
        case 31: \
            dfp[search_index * PASSWORD_LENGTH + 30] = (b7.s##suffix >> 16) & 0xff; \
        case 30: \
            dfp[search_index * PASSWORD_LENGTH + 29] = (b7.s##suffix >> 8) & 0xff; \
        case 29: \
            dfp[search_index * PASSWORD_LENGTH + 28] = (b7.s##suffix >> 0) & 0xff; \
        case 28: \
            dfp[search_index * PASSWORD_LENGTH + 27] = (b6.s##suffix >> 24) & 0xff; \
        case 27: \
            dfp[search_index * PASSWORD_LENGTH + 26] = (b6.s##suffix >> 16) & 0xff; \
        case 26: \
            dfp[search_index * PASSWORD_LENGTH + 25] = (b6.s##suffix >> 8) & 0xff; \
        case 25: \
            dfp[search_index * PASSWORD_LENGTH + 24] = (b6.s##suffix >> 0) & 0xff; \
        case 24: \
            dfp[search_index * PASSWORD_LENGTH + 23] = (b5.s##suffix >> 24) & 0xff; \
        case 23: \
            dfp[search_index * PASSWORD_LENGTH + 22] = (b5.s##suffix >> 16) & 0xff; \
        case 22: \
            dfp[search_index * PASSWORD_LENGTH + 21] = (b5.s##suffix >> 8) & 0xff; \
        case 21: \
            dfp[search_index * PASSWORD_LENGTH + 20] = (b5.s##suffix >> 0) & 0xff; \
        case 20: \
            dfp[search_index * PASSWORD_LENGTH + 19] = (b4.s##suffix >> 24) & 0xff; \
        case 19: \
            dfp[search_index * PASSWORD_LENGTH + 18] = (b4.s##suffix >> 16) & 0xff; \
        case 18: \
            dfp[search_index * PASSWORD_LENGTH + 17] = (b4.s##suffix >> 8) & 0xff; \
        case 17: \
            dfp[search_index * PASSWORD_LENGTH + 16] = (b4.s##suffix >> 0) & 0xff; \
        case 16: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b3.s##suffix >> 24) & 0xff; \
        case 15: \
            dfp[search_index * PASSWORD_LENGTH + 14] = (b3.s##suffix >> 16) & 0xff; \
        case 14: \
            dfp[search_index * PASSWORD_LENGTH + 13] = (b3.s##suffix >> 8) & 0xff; \
        case 13: \
            dfp[search_index * PASSWORD_LENGTH + 12] = (b3.s##suffix >> 0) & 0xff; \
        case 12: \
            dfp[search_index * PASSWORD_LENGTH + 11] = (b2.s##suffix >> 24) & 0xff; \
        case 11: \
            dfp[search_index * PASSWORD_LENGTH + 10] = (b2.s##suffix >> 16) & 0xff; \
        case 10: \
            dfp[search_index * PASSWORD_LENGTH + 9] = (b2.s##suffix >> 8) & 0xff; \
        case 9: \
            dfp[search_index * PASSWORD_LENGTH + 8] = (b2.s##suffix >> 0) & 0xff; \
        case 8: \
            dfp[search_index * PASSWORD_LENGTH + 7] = (b1.s##suffix >> 24) & 0xff; \
        case 7: \
            dfp[search_index * PASSWORD_LENGTH + 6] = (b1.s##suffix >> 16) & 0xff; \
        case 6: \
            dfp[search_index * PASSWORD_LENGTH + 5] = (b1.s##suffix >> 8) & 0xff; \
        case 5: \
            dfp[search_index * PASSWORD_LENGTH + 4] = (b1.s##suffix >> 0) & 0xff; \
        case 4: \
            dfp[search_index * PASSWORD_LENGTH + 3] = (b0.s##suffix >> 24) & 0xff; \
        case 3: \
            dfp[search_index * PASSWORD_LENGTH + 2] = (b0.s##suffix >> 16) & 0xff; \
        case 2: \
            dfp[search_index * PASSWORD_LENGTH + 1] = (b0.s##suffix >> 8) & 0xff; \
        case 1: \
            dfp[search_index * PASSWORD_LENGTH + 0] = (b0.s##suffix >> 0) & 0xff; \
    } \
    dfpf[search_index] = (unsigned char) 1; \
}
#endif

/**
 * Big endian copy from b0-b13, for lengths 0-55, MD5-style data with all
 * the bytes in each word filled with a character (as opposed to NTLM).  Used
 * for SHA1, SHA256, etc.
 */


// dfp: Device Found Passwords
// dfpf: Device Found Passwords Flags
#ifdef grt_vector_1
#define CopyFoundPasswordToMemoryBE(dfp, dfpf, suffix) { \
    switch ( PASSWORD_LENGTH ) { \
        case 55: \
            dfp[search_index * PASSWORD_LENGTH + 54] = (b13 >> 8) & 0xff; \
        case 54: \
            dfp[search_index * PASSWORD_LENGTH + 53] = (b13 >> 16) & 0xff; \
        case 53: \
            dfp[search_index * PASSWORD_LENGTH + 52] = (b13 >> 24) & 0xff; \
        case 52: \
            dfp[search_index * PASSWORD_LENGTH + 51] = (b12 >> 0) & 0xff; \
        case 51: \
            dfp[search_index * PASSWORD_LENGTH + 50] = (b12 >> 8) & 0xff; \
        case 50: \
            dfp[search_index * PASSWORD_LENGTH + 49] = (b12 >> 16) & 0xff; \
        case 49: \
            dfp[search_index * PASSWORD_LENGTH + 48] = (b12 >> 24) & 0xff; \
        case 48: \
            dfp[search_index * PASSWORD_LENGTH + 47] = (b11 >> 0) & 0xff; \
        case 47: \
            dfp[search_index * PASSWORD_LENGTH + 46] = (b11 >> 8) & 0xff; \
        case 46: \
            dfp[search_index * PASSWORD_LENGTH + 45] = (b11 >> 16) & 0xff; \
        case 45: \
            dfp[search_index * PASSWORD_LENGTH + 44] = (b11 >> 24) & 0xff; \
        case 44: \
            dfp[search_index * PASSWORD_LENGTH + 43] = (b10 >> 0) & 0xff; \
        case 43: \
            dfp[search_index * PASSWORD_LENGTH + 42] = (b10 >> 8) & 0xff; \
        case 42: \
            dfp[search_index * PASSWORD_LENGTH + 41] = (b10 >> 16) & 0xff; \
        case 41: \
            dfp[search_index * PASSWORD_LENGTH + 40] = (b10 >> 24) & 0xff; \
        case 40: \
            dfp[search_index * PASSWORD_LENGTH + 39] = (b9 >> 0) & 0xff; \
        case 39: \
            dfp[search_index * PASSWORD_LENGTH + 38] = (b9 >> 8) & 0xff; \
        case 38: \
            dfp[search_index * PASSWORD_LENGTH + 37] = (b9 >> 16) & 0xff; \
        case 37: \
            dfp[search_index * PASSWORD_LENGTH + 36] = (b9 >> 24) & 0xff; \
        case 36: \
            dfp[search_index * PASSWORD_LENGTH + 35] = (b8 >> 0) & 0xff; \
        case 35: \
            dfp[search_index * PASSWORD_LENGTH + 34] = (b8 >> 8) & 0xff; \
        case 34: \
            dfp[search_index * PASSWORD_LENGTH + 33] = (b8 >> 16) & 0xff; \
        case 33: \
            dfp[search_index * PASSWORD_LENGTH + 32] = (b8 >> 24) & 0xff; \
        case 32: \
            dfp[search_index * PASSWORD_LENGTH + 31] = (b7 >> 0) & 0xff; \
        case 31: \
            dfp[search_index * PASSWORD_LENGTH + 30] = (b7 >> 8) & 0xff; \
        case 30: \
            dfp[search_index * PASSWORD_LENGTH + 29] = (b7 >> 16) & 0xff; \
        case 29: \
            dfp[search_index * PASSWORD_LENGTH + 28] = (b7 >> 24) & 0xff; \
        case 28: \
            dfp[search_index * PASSWORD_LENGTH + 27] = (b6 >> 0) & 0xff; \
        case 27: \
            dfp[search_index * PASSWORD_LENGTH + 26] = (b6 >> 8) & 0xff; \
        case 26: \
            dfp[search_index * PASSWORD_LENGTH + 25] = (b6 >> 16) & 0xff; \
        case 25: \
            dfp[search_index * PASSWORD_LENGTH + 24] = (b6 >> 24) & 0xff; \
        case 24: \
            dfp[search_index * PASSWORD_LENGTH + 23] = (b5 >> 0) & 0xff; \
        case 23: \
            dfp[search_index * PASSWORD_LENGTH + 22] = (b5 >> 8) & 0xff; \
        case 22: \
            dfp[search_index * PASSWORD_LENGTH + 21] = (b5 >> 16) & 0xff; \
        case 21: \
            dfp[search_index * PASSWORD_LENGTH + 20] = (b5 >> 24) & 0xff; \
        case 20: \
            dfp[search_index * PASSWORD_LENGTH + 19] = (b4 >> 0) & 0xff; \
        case 19: \
            dfp[search_index * PASSWORD_LENGTH + 18] = (b4 >> 8) & 0xff; \
        case 18: \
            dfp[search_index * PASSWORD_LENGTH + 17] = (b4 >> 16) & 0xff; \
        case 17: \
            dfp[search_index * PASSWORD_LENGTH + 16] = (b4 >> 24) & 0xff; \
        case 16: \
            dfp[search_index * PASSWORD_LENGTH + 15] = (b3 >> 0) & 0xff; \
        case 15: \
            dfp[search_index * PASSWORD_LENGTH + 14] = (b3 >> 8) & 0xff; \
        case 14: \
            dfp[search_index * PASSWORD_LENGTH + 13] = (b3 >> 16) & 0xff; \
        case 13: \
            dfp[search_index * PASSWORD_LENGTH + 12] = (b3 >> 24) & 0xff; \
        case 12: \
            dfp[search_index * PASSWORD_LENGTH + 11] = (b2 >> 0) & 0xff; \
        case 11: \
            dfp[search_index * PASSWORD_LENGTH + 10] = (b2 >> 8) & 0xff; \
        case 10: \
            dfp[search_index * PASSWORD_LENGTH + 9] = (b2 >> 16) & 0xff; \
        case 9: \
            dfp[search_index * PASSWORD_LENGTH + 8] = (b2 >> 24) & 0xff; \
        case 8: \
            dfp[search_index * PASSWORD_LENGTH + 7] = (b1 >> 0) & 0xff; \
        case 7: \
            dfp[search_index * PASSWORD_LENGTH + 6] = (b1 >> 8) & 0xff; \
        case 6: \
            dfp[search_index * PASSWORD_LENGTH + 5] = (b1 >> 16) & 0xff; \
        case 5: \
            dfp[search_index * PASSWORD_LENGTH + 4] = (b1 >> 24) & 0xff; \
        case 4: \
            dfp[search_index * PASSWORD_LENGTH + 3] = (b0 >> 0) & 0xff; \
        case 3: \
            dfp[search_index * PASSWORD_LENGTH + 2] = (b0 >> 8) & 0xff; \
        case 2: \
            dfp[search_index * PASSWORD_LENGTH + 1] = (b0 >> 16) & 0xff; \
        case 1: \
            dfp[search_index * PASSWORD_LENGTH + 0] = (b0 >> 24) & 0xff; \
    } \
    dfpf[search_index] = (unsigned char) 1; \
}
#else
#define CopyFoundPasswordToMemoryBE(dfp, dfpf, suffix) { \
    switch ( PASSWORD_LENGTH ) { \
        case 55: \
            dfp[search_index * PASSWORD_LENGTH + 54] = (b13.s##suffix >> 8) & 0xff; \
        case 54: \
            dfp[search_index * PASSWORD_LENGTH + 53] = (b13.s##suffix >> 16) & 0xff; \
        case 53: \
            dfp[search_index * PASSWORD_LENGTH + 52] = (b13.s##suffix >> 24) & 0xff; \
        case 52: \
            dfp[search_index * PASSWORD_LENGTH + 51] = (b12.s##suffix >> 0) & 0xff; \
        case 51: \
            dfp[search_index * PASSWORD_LENGTH + 50] = (b12.s##suffix >> 8) & 0xff; \
        case 50: \
            dfp[search_index * PASSWORD_LENGTH + 49] = (b12.s##suffix >> 16) & 0xff; \
        case 49: \
            dfp[search_index * PASSWORD_LENGTH + 48] = (b12.s##suffix >> 24) & 0xff; \
        case 48: \
            dfp[search_index * PASSWORD_LENGTH + 47] = (b11.s##suffix >> 0) & 0xff; \
        case 47: \
            dfp[search_index * PASSWORD_LENGTH + 46] = (b11.s##suffix >> 8) & 0xff; \
        case 46: \
            dfp[search_index * PASSWORD_LENGTH + 45] = (b11.s##suffix >> 16) & 0xff; \
        case 45: \
            dfp[search_index * PASSWORD_LENGTH + 44] = (b11.s##suffix >> 24) & 0xff; \
        case 44: \
            dfp[search_index * PASSWORD_LENGTH + 43] = (b10.s##suffix >> 0) & 0xff; \
        case 43: \
            dfp[search_index * PASSWORD_LENGTH + 42] = (b10.s##suffix >> 8) & 0xff; \
        case 42: \
            dfp[search_index * PASSWORD_LENGTH + 41] = (b10.s##suffix >> 16) & 0xff; \
        case 41: \
            dfp[search_index * PASSWORD_LENGTH + 40] = (b10.s##suffix >> 24) & 0xff; \
        case 40: \
            dfp[search_index * PASSWORD_LENGTH + 39] = (b9.s##suffix >> 0) & 0xff; \
        case 39: \
            dfp[search_index * PASSWORD_LENGTH + 38] = (b9.s##suffix >> 8) & 0xff; \
        case 38: \
            dfp[search_index * PASSWORD_LENGTH + 37] = (b9.s##suffix >> 16) & 0xff; \
        case 37: \
            dfp[search_index * PASSWORD_LENGTH + 36] = (b9.s##suffix >> 24) & 0xff; \
        case 36: \
            dfp[search_index * PASSWORD_LENGTH + 35] = (b8.s##suffix >> 0) & 0xff; \
        case 35: \
            dfp[search_index * PASSWORD_LENGTH + 34] = (b8.s##suffix >> 8) & 0xff; \
        case 34: \
            dfp[search_index * PASSWORD_LENGTH + 33] = (b8.s##suffix >> 16) & 0xff; \
        case 33: \
            dfp[search_index * PASSWORD_LENGTH + 32] = (b8.s##suffix >> 24) & 0xff; \
        case 32: \
            dfp[search_index * PASSWORD_LENGTH + 31] = (b7.s##suffix >> 0) & 0xff; \
        case 31: \
            dfp[search_index * PASSWORD_LENGTH + 30] = (b7.s##suffix >> 8) & 0xff; \
        case 30: \
            dfp[search_index * PASSWORD_LENGTH + 29] = (b7.s##suffix >> 16) & 0xff; \
        case 29: \
            dfp[search_index * PASSWORD_LENGTH + 28] = (b7.s##suffix >> 24) & 0xff; \
        case 28: \
            dfp[search_index * PASSWORD_LENGTH + 27] = (b6.s##suffix >> 0) & 0xff; \
        case 27: \
            dfp[search_index * PASSWORD_LENGTH + 26] = (b6.s##suffix >> 8) & 0xff; \
        case 26: \
            dfp[search_index * PASSWORD_LENGTH + 25] = (b6.s##suffix >> 16) & 0xff; \
        case 25: \
            dfp[search_index * PASSWORD_LENGTH + 24] = (b6.s##suffix >> 24) & 0xff; \
        case 24: \
            dfp[search_index * PASSWORD_LENGTH + 23] = (b5.s##suffix >> 0) & 0xff; \
        case 23: \
            dfp[search_index * PASSWORD_LENGTH + 22] = (b5.s##suffix >> 8) & 0xff; \
        case 22: \
            dfp[search_index * PASSWORD_LENGTH + 21] = (b5.s##suffix >> 16) & 0xff; \
        case 21: \
            dfp[search_index * PASSWORD_LENGTH + 20] = (b5.s##suffix >> 24) & 0xff; \
        case 20: \
            dfp[search_index * PASSWORD_LENGTH + 19] = (b4.s##suffix >> 0) & 0xff; \
        case 19: \
            dfp[search_index * PASSWORD_LENGTH + 18] = (b4.s##suffix >> 8) & 0xff; \
        case 18: \
            dfp[search_index * PASSWORD_LENGTH + 17] = (b4.s##suffix >> 16) & 0xff; \
        case 17: \
            dfp[search_index * PASSWORD_LENGTH + 16] = (b4.s##suffix >> 24) & 0xff; \
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
    dfpf[search_index] = (unsigned char) 1; \
}
#endif


#ifdef grt_vector_1
#define CheckPassword128LE(dgh, dfp, dfpf, dnh, suffix) { \
    search_high = dnh; \
    search_low = 0; \
    while (search_low < search_high) { \
        search_index = search_low + (search_high - search_low) / 2; \
        current_hash_value = dgh[4 * search_index]; \
        if (current_hash_value < a) { \
            search_low = search_index + 1; \
        } else { \
            search_high = search_index; \
        } \
        if ((a == current_hash_value) && (search_low < dnh)) { \
            break; \
        } \
    } \
    if (a == current_hash_value) { \
        while (search_index && (a == dgh[(search_index - 1) * 4])) { \
            search_index--; \
        } \
        while ((a == dgh[search_index * 4])) { \
            if (b == dgh[search_index * 4 + 1]) { \
                if (c == dgh[search_index * 4 + 2]) { \
                    if (d == dgh[search_index * 4 + 3]) { \
                    CopyFoundPasswordToMemoryLE(dfp, dfpf, suffix); \
                    } \
                } \
            } \
            search_index++; \
        } \
    } \
}
#else
#define CheckPassword128LE(dgh, dfp, dfpf, dnh, suffix) { \
    search_high = dnh; \
    search_low = 0; \
    while (search_low < search_high) { \
        search_index = search_low + (search_high - search_low) / 2; \
        current_hash_value = dgh[4 * search_index]; \
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
        while (search_index && (a.s##suffix == dgh[(search_index - 1) * 4])) { \
            search_index--; \
        } \
        while ((a.s##suffix == dgh[search_index * 4])) { \
            if (b.s##suffix == dgh[search_index * 4 + 1]) { \
                if (c.s##suffix == dgh[search_index * 4 + 2]) { \
                    if (d.s##suffix == dgh[search_index * 4 + 3]) { \
                    CopyFoundPasswordToMemoryLE(dfp, dfpf, suffix); \
                    } \
                } \
            } \
            search_index++; \
        } \
    } \
}
#endif

#ifdef grt_vector_1
#define CheckPassword256BE(dgh, dfp, dfpf, dnh, suffix) { \
    search_high = dnh; \
    search_low = 0; \
    while (search_low < search_high) { \
        search_index = search_low + (search_high - search_low) / 2; \
        current_hash_value = dgh[8 * search_index]; \
        if (current_hash_value < a) { \
            search_low = search_index + 1; \
        } else { \
            search_high = search_index; \
        } \
        if ((a == current_hash_value) && (search_low < dnh)) { \
            break; \
        } \
    } \
    if (a == current_hash_value) { \
        while (search_index && (a == dgh[(search_index - 1) * 8])) { \
            search_index--; \
        } \
        while ((a == dgh[search_index * 8])) { \
            if (b == dgh[search_index * 8 + 1]) { \
                if (c == dgh[search_index * 8 + 2]) { \
                    if (d == dgh[search_index * 8 + 3]) { \
                    CopyFoundPasswordToMemoryBE(dfp, dfpf, suffix); \
                    } \
                } \
            } \
            search_index++; \
        } \
    } \
}
#else
#define CheckPassword256BE(dgh, dfp, dfpf, dnh, suffix) { \
    search_high = dnh; \
    search_low = 0; \
    while (search_low < search_high) { \
        search_index = search_low + (search_high - search_low) / 2; \
        current_hash_value = dgh[8 * search_index]; \
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
        while (search_index && (a.s##suffix == dgh[(search_index - 1) * 8])) { \
            search_index--; \
        } \
        while ((a.s##suffix == dgh[search_index * 8])) { \
            if (b.s##suffix == dgh[search_index * 8 + 1]) { \
                if (c.s##suffix == dgh[search_index * 8 + 2]) { \
                    if (d.s##suffix == dgh[search_index * 8 + 3]) { \
                    CopyFoundPasswordToMemoryBE(dfp, dfpf, suffix); \
                    } \
                } \
            } \
            search_index++; \
        } \
    } \
}
#endif

