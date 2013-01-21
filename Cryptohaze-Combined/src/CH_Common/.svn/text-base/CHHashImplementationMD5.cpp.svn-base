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
 * To unit test, you can build it from src/CH_Common with something like:
 * g++ -o /tmp/md5test CHHashImplementation.cpp CHHashImplementationMD5.cpp \
 *   -I ../../inc -DUNIT_TEST
 */

#include <CH_Common/CHHashImplementationMD5.h>

#include <CH_HashDefines/CH_MD5.h>
#include <string.h>
#include <stdio.h>

std::vector<uint8_t> CHHashImplementationMD5::hashData(
    const std::vector<uint8_t> &rawData) {
    // Do a forward MD5 hash.
    
    std::vector<uint8_t> rawHashVector;
    uint32_t blocksToRun;
    
    // Size the block data vector
    rawHashVector = rawData;
    //printf("rawHashVector initial size: %d\n", rawHashVector.size());
    // Check to see if an extra block is needed - as long as 55 or fewer bytes
    // are in the final block, that can be the final block, else another block
    // is needed.
    if ((rawData.size() % 64) > 55) {
        //printf("Extra block needed!");
        // Add another block - 64 bytes
        rawHashVector.resize(rawData.size() + 64, 0);
    }
    // Round up to 64 bytes
    rawHashVector.resize(rawHashVector.size() +
        (64 - (rawHashVector.size() % 64)), 0);
    
    //printf("rawHashVector.size(): %d\n", rawHashVector.size());
    
    // Set the padding bit
    rawHashVector[rawData.size()] = 0x80;
    
    blocksToRun = rawHashVector.size() / 64;
    //printf("Blocks to run: %d\n", blocksToRun);
    
    
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14,
            b15, a, b, c, d, prev_a, prev_b, prev_c, prev_d;
    
    uint32_t *rawHashWords;
        
    // Clear all variables.
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = b13 =
            b14 = b15 = 0;
    
    // Set the length
    rawHashWords = (uint32_t *)&rawHashVector[rawHashVector.size() - 8];
    *rawHashWords = (rawData.size() * 8);
    
    // Do block 0
    rawHashWords = (uint32_t *)&rawHashVector[0];
    
    b0  = rawHashWords[ 0];
    b1  = rawHashWords[ 1];
    b2  = rawHashWords[ 2];
    b3  = rawHashWords[ 3];
    b4  = rawHashWords[ 4];
    b5  = rawHashWords[ 5];
    b6  = rawHashWords[ 6];
    b7  = rawHashWords[ 7];
    b8  = rawHashWords[ 8];
    b9  = rawHashWords[ 9];
    b10 = rawHashWords[10];
    b11 = rawHashWords[11];
    b12 = rawHashWords[12];
    b13 = rawHashWords[13];
    b14 = rawHashWords[14];
    b15 = rawHashWords[15];
    
    MD5_FULL_HASH();
    
    for (int i = 1; i < blocksToRun; i++ ) {
        prev_a = a;
        prev_b = b;
        prev_c = c;
        prev_d = d;
        rawHashWords = (uint32_t *)&rawHashVector[64 * i];
        b0  = rawHashWords[ 0];
        b1  = rawHashWords[ 1];
        b2  = rawHashWords[ 2];
        b3  = rawHashWords[ 3];
        b4  = rawHashWords[ 4];
        b5  = rawHashWords[ 5];
        b6  = rawHashWords[ 6];
        b7  = rawHashWords[ 7];
        b8  = rawHashWords[ 8];
        b9  = rawHashWords[ 9];
        b10 = rawHashWords[10];
        b11 = rawHashWords[11];
        b12 = rawHashWords[12];
        b13 = rawHashWords[13];
        b14 = rawHashWords[14];
        b15 = rawHashWords[15];
        MD5_FULL_HASH_SECOND_ROUND(prev_a, prev_b, prev_c, prev_d);
    }
    
    // Copy and return results.
    std::vector<uint8_t> returnHash;
    returnHash.resize(16, 0);
    uint32_t *returnHashWords = (uint32_t *)&returnHash[0];
    returnHashWords[0] = a;
    returnHashWords[1] = b;
    returnHashWords[2] = c;
    returnHashWords[3] = d;
    return returnHash;
}

void CHHashImplementationMD5::prepareHash(int passLength,
        std::vector<uint8_t> &rawHash) {

    uint32_t a, b, c, d;
    uint32_t *hash32 = (uint32_t *)&rawHash[0];

    a = hash32[0];
    b = hash32[1];
    c = hash32[2];
    d = hash32[3];
    
    a -= 0x67452301;
    b -= 0xefcdab89;
    c -= 0x98badcfe;
    d -= 0x10325476;
    
    if (passLength < 8) {
        REV_II (b, c, d, a, 0x00 /*b9*/, MD5S44, 0xeb86d391); //64
        REV_II (c, d, a, b, 0x00 /*b2*/, MD5S43, 0x2ad7d2bb); //63
        REV_II (d, a, b, c, 0x00 /*b11*/, MD5S42, 0xbd3af235); //62
        REV_II (a, b, c, d, 0x00 /*b4*/, MD5S41, 0xf7537e82); //61
        REV_II (b, c, d, a, 0x00 /*b13*/, MD5S44, 0x4e0811a1); //60
        REV_II (c, d, a, b, 0x00 /*b6*/, MD5S43, 0xa3014314); //59
        REV_II (d, a, b, c, 0x00 /*b15*/, MD5S42, 0xfe2ce6e0); //58
        REV_II (a, b, c, d, 0x00 /*b8*/, MD5S41, 0x6fa87e4f); //57
    } else if (passLength == 8) {
        REV_II (b, c, d, a, 0x00 /*b9*/, MD5S44, 0xeb86d391); //64
        // Padding bit will be set
        REV_II (c, d, a, b, 0x00000080 /*b2*/, MD5S43, 0x2ad7d2bb); //63
        REV_II (d, a, b, c, 0x00 /*b11*/, MD5S42, 0xbd3af235); //62
        REV_II (a, b, c, d, 0x00 /*b4*/, MD5S41, 0xf7537e82); //61
        REV_II (b, c, d, a, 0x00 /*b13*/, MD5S44, 0x4e0811a1); //60
        REV_II (c, d, a, b, 0x00 /*b6*/, MD5S43, 0xa3014314); //59
        REV_II (d, a, b, c, 0x00 /*b15*/, MD5S42, 0xfe2ce6e0); //58
        REV_II (a, b, c, d, 0x00 /*b8*/, MD5S41, 0x6fa87e4f); //57
    }
    
    hash32[0] = a;
    hash32[1] = b;
    hash32[2] = c;
    hash32[3] = d;
}

void CHHashImplementationMD5::postProcessHash(int passLength,
        std::vector<uint8_t> &rawHash) {
    
    uint32_t a, b, c, d;
    uint32_t *hash32 = (uint32_t *)&rawHash[0];
    
    a = hash32[0];
    b = hash32[1];
    c = hash32[2];
    d = hash32[3];

    if (passLength < 8) {
        MD5II(a, b, c, d, 0x00, MD5S41, 0x6fa87e4f); /* 57 */
        MD5II(d, a, b, c, 0x00, MD5S42, 0xfe2ce6e0); /* 58 */
        MD5II(c, d, a, b, 0x00, MD5S43, 0xa3014314); /* 59 */
        MD5II(b, c, d, a, 0x00, MD5S44, 0x4e0811a1); /* 60 */
        MD5II(a, b, c, d, 0x00, MD5S41, 0xf7537e82); /* 61 */
        MD5II(d, a, b, c, 0x00, MD5S42, 0xbd3af235); /* 62 */
        MD5II(c, d, a, b, 0x00, MD5S43, 0x2ad7d2bb); /* 63 */
        MD5II(b, c, d, a, 0x00, MD5S44, 0xeb86d391); /* 64 */
    } else if (passLength == 8) {
        MD5II(a, b, c, d, 0x00, MD5S41, 0x6fa87e4f); /* 57 */
        MD5II(d, a, b, c, 0x00, MD5S42, 0xfe2ce6e0); /* 58 */
        MD5II(c, d, a, b, 0x00, MD5S43, 0xa3014314); /* 59 */
        MD5II(b, c, d, a, 0x00, MD5S44, 0x4e0811a1); /* 60 */
        MD5II(a, b, c, d, 0x00, MD5S41, 0xf7537e82); /* 61 */
        MD5II(d, a, b, c, 0x00, MD5S42, 0xbd3af235); /* 62 */
        MD5II(c, d, a, b, 0x00000080, MD5S43, 0x2ad7d2bb); /* 63 */
        MD5II(b, c, d, a, 0x00, MD5S44, 0xeb86d391); /* 64 */
    }
    
    a += 0x67452301;
    b += 0xefcdab89;
    c += 0x98badcfe;
    d += 0x10325476;
    
    hash32[0] = a;
    hash32[1] = b;
    hash32[2] = c;
    hash32[3] = d;
}

#ifdef UNIT_TEST

#include <stdio.h>
#include <string>

uint8_t nullDataHash[] = {0xd4, 0x1d, 0x8c, 0xd9, 0x8f, 0x00, 0xb2, 0x04, 
                          0xe9, 0x80, 0x09, 0x98, 0xec, 0xf8, 0x42, 0x7e};

uint8_t singleAHash[] = {0x0c, 0xc1, 0x75, 0xb9, 0xc0, 0xf1, 0xb6, 0xa8, 
                         0x31, 0xc3, 0x99, 0xe2, 0x69, 0x77, 0x26, 0x61};

uint8_t abcHash[] = {0x90, 0x01, 0x50, 0x98, 0x3c, 0xd2, 0x4f, 0xb0,
                     0xd6, 0x96, 0x3f, 0x7d, 0x28, 0xe1, 0x7f, 0x72};

uint8_t mdHash[] = {0xf9, 0x6b, 0x69, 0x7d, 0x7c, 0xb7, 0x93, 0x8d,
                    0x52, 0x5a, 0x2f, 0x31, 0xaa, 0xf1, 0x61, 0xd0};

uint8_t laHash[] = {0xc3, 0xfc, 0xd3, 0xd7, 0x61, 0x92, 0xe4, 0x00,
                    0x7d, 0xfb, 0x49, 0x6c, 0xca, 0x67, 0xe1, 0x3b};

// Lower Upper Numeric
uint8_t lunHash[] = {0xd1, 0x74, 0xab, 0x98, 0xd2, 0x77, 0xd9, 0xf5,
                     0xa5, 0x61, 0x1c, 0x2c, 0x9f, 0x41, 0x9d, 0x9f};

// Long Numeric
uint8_t lnHash[] = {0x57, 0xed, 0xf4, 0xa2, 0x2b, 0xe3, 0xc9, 0x55,
                    0xac, 0x49, 0xda, 0x2e, 0x21, 0x07, 0xb6, 0x7a};

void printHash(std::vector<uint8_t> &hash) {
    for (int i = 0; i < hash.size(); i++) {
        printf("%02x", hash[i]);
    }
    printf("\n");
}

int main() {
    CHHashImplementationMD5 MD5Test;
    std::vector<uint8_t> hash;
    std::string hashString;

    // Test null hash
    std::vector<uint8_t> nullData;
    
    hash = MD5Test.hashData(nullData);
    hashString = MD5Test.hashDataAsciiString(nullData);
    
    printf("Null hash: ");
    printHash(hash);
    printf("Null hash: ");
    printf("%s\n", hashString.c_str());
    
    // Test the null hash
    if (memcmp(nullDataHash, &hash[0], 16) != 0) {
        printf("ASSERT FAILED FOR NULL HASH!\n");
    }
    
    // Test 'a' hash
    std::vector<uint8_t> aData;
    aData.push_back('a');
    
    hash = MD5Test.hashData(aData);
    hashString = MD5Test.hashDataAsciiString(aData);
    
    printf("\n");
    printf("'a' hash: ");
    printHash(hash);
    printf("'a' hash: ");
    printf("%s\n", hashString.c_str());
    if (memcmp(singleAHash, &hash[0], 16) != 0) {
        printf("ASSERT FAILED FOR 'a' HASH!\n");
    }

    // Test 'abc' hash
    std::vector<uint8_t> abcData;
    abcData.push_back('a');
    abcData.push_back('b');
    abcData.push_back('c');
    
    hash = MD5Test.hashData(abcData);
    hashString = MD5Test.hashDataAsciiString(abcData);
   
    printf("\n");
    printf("'abc' hash: ");
    printHash(hash);
    printf("'abc' hash: ");
    printf("%s\n", hashString.c_str());
    
    if (memcmp(abcHash, &hash[0], 16) != 0) {
        printf("ASSERT FAILED FOR 'abc' HASH!\n");
    }
    
    std::string mdString("message digest");
    std::vector<uint8_t> mdData = std::vector<uint8_t>(mdString.c_str(),
        mdString.c_str() + mdString.length());

    hash = MD5Test.hashData(mdData);
    hashString = MD5Test.hashDataAsciiString(mdData);
    
    printf("\n");
    printf("'message digest' hash: ");
    printHash(hash);
    printf("'message digest' hash: ");
    printf("%s\n", hashString.c_str());
    
    if (memcmp(mdHash, &hash[0], 16) != 0) {
        printf("ASSERT FAILED FOR 'message digest' HASH!\n");
    }

    // lower alpha
    std::string laString("abcdefghijklmnopqrstuvwxyz");
    std::vector<uint8_t> laData = std::vector<uint8_t>(laString.c_str(),
        laString.c_str() + laString.length());

    hash = MD5Test.hashData(laData);
    hashString = MD5Test.hashDataAsciiString(laData);

    printf("\n");
    printf("loweralpha hash: ");
    printHash(hash);
    printf("loweralpha hash: ");
    printf("%s\n", hashString.c_str());
    
    if (memcmp(laHash, &hash[0], 16) != 0) {
        printf("ASSERT FAILED FOR loweralpha HASH!\n");
    }
    
    
    // Lower Upper Numeric
    std::string lunString("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789");
    std::vector<uint8_t> lunData = std::vector<uint8_t>(lunString.c_str(),
        lunString.c_str() + lunString.length());

    hash = MD5Test.hashData(lunData);
    hashString = MD5Test.hashDataAsciiString(lunData);
    
    printf("\n");
    printf("Lower Upper Numeric hash: ");
    printHash(hash);
    printf("Lower Upper Numeric hash: ");
    printf("%s\n", hashString.c_str());
    
    if (memcmp(lunHash, &hash[0], 16) != 0) {
        printf("ASSERT FAILED FOR LUN HASH!\n");
    }

    // Long Numeric
    std::string lnString("12345678901234567890123456789012345678901234567890123456789012345678901234567890");
    std::vector<uint8_t> lnData = std::vector<uint8_t>(lnString.c_str(),
        lnString.c_str() + lnString.length());

    hash = MD5Test.hashData(lnData);
    hashString = MD5Test.hashDataAsciiString(lnData);
    
    printf("\n");
    printf("Long Numeric hash: ");
    printHash(hash);
    printf("Long Numeric hash: ");
    printf("%s\n", hashString.c_str());
    
    if (memcmp(lnHash, &hash[0], 16) != 0) {
        printf("ASSERT FAILED FOR LN HASH!\n");
    }
    
    printf("\n\nTesting vector hashing...\n");
    std::vector<std::vector<uint8_t> > dataToHash;
    std::vector<std::string> hashedStrings;
    
    dataToHash.push_back(nullData);
    dataToHash.push_back(aData);
    dataToHash.push_back(abcData);
    dataToHash.push_back(mdData);
    dataToHash.push_back(laData);
    dataToHash.push_back(lunData);
    dataToHash.push_back(lnData);
    
    hashedStrings = MD5Test.hashMultipleDataAsciiString(dataToHash, 0);

    for (int i = 0; i < hashedStrings.size(); i++) {
        printf("%s\n", hashedStrings[i].c_str());
    }
    
    // Do a quick test of correctness over a long stretch.
    printf("\n\nDoing final hash correctness test...\n");
    std::string hashesSum;
    std::vector<uint8_t> longHashesVector;
    std::vector<uint8_t> longHashData;
    for (int i = 0; i < 10000; i++) {
        std::vector<uint8_t> hashResult;
        longHashData.push_back('1');
        hashResult = MD5Test.hashDataAsciiVector(longHashData);
        longHashesVector.insert(longHashesVector.end(), hashResult.begin(), hashResult.end());
    }
    hashesSum = MD5Test.hashDataAsciiString(longHashesVector);
    printf("Final hash sum: %s\n", hashesSum.c_str());
    printf("CORRECT sum   : ea53bad8459f4a9499449720ee1bbe3e\n");

    return 0;
}
#endif