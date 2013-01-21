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

#include "CH_HashFiles/CHHashFileV.h"
#include "MFN_Common/MFNDefines.h"

#include <vector>
#include <string>
#include <stdlib.h>

static inline bool is_base64(unsigned char c) {
  return (isalnum(c) || (c == '+') || (c == '/'));
}

// Default, bog standard base64 charsets for different functions.
static std::string defaultBase64Charset =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static std::string defaultPHPBBCharset =
    "./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

std::vector<uint8_t> CHHashFileV::convertAsciiToBinary(std::string asciiHex) {
    std::vector<uint8_t> returnVector;
    char convertSpace[3];
    uint32_t result;

    // Check for even length - if not even, return null vector.
    if (asciiHex.length() % 2) {
        return returnVector;
    }

    // Loop until either maxLength is hit, or strlen(intput) / 2 is hit.
    for (size_t i = 0; i < (asciiHex.length() / 2); i++) {
        convertSpace[0] = asciiHex[2 * i];
        convertSpace[1] = asciiHex[2 * i + 1];
        convertSpace[2] = 0;
        sscanf(convertSpace, "%2x", &result);
        // Do this to prevent scanf from overwriting memory with a 4 byte value...
        returnVector.push_back((uint8_t) result & 0xff);
    }
    return returnVector;
}


std::vector<uint8_t> CHHashFileV::convertAsciiToBinary(std::vector<char> asciiHex) {
    std::vector<uint8_t> returnVector;
    char convertSpace[3];
    uint32_t result;

    // Check for even length - if not even, return null vector.
    if (asciiHex.size() % 2) {
        return returnVector;
    }

    // Loop until either maxLength is hit, or strlen(intput) / 2 is hit.
    for (size_t i = 0; i < (asciiHex.size() / 2); i++) {
        convertSpace[0] = asciiHex[2 * i];
        convertSpace[1] = asciiHex[2 * i + 1];
        convertSpace[2] = 0;
        sscanf(convertSpace, "%2x", &result);
        // Do this to prevent scanf from overwriting memory with a 4 byte value...
        returnVector.push_back((uint8_t) result & 0xff);
    }
    return returnVector;
}

std::vector<uint8_t> CHHashFileV::base64Encode(
        std::vector<uint8_t> bytesToEncode,
        std::string base64Characters) {

    std::vector<uint8_t> encodedData;

    int i = 0;
    int j = 0;
    uint8_t char_array_3[3];
    uint8_t char_array_4[4];
    
    // How much data to encode
    uint32_t in_len = bytesToEncode.size();
    std::vector<uint8_t>::iterator bytes_to_encode = bytesToEncode.begin();

    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;
            for (i = 0; (i < 4); i++) {
                encodedData.push_back(base64Characters[char_array_4[i]]);
            }
            memset(char_array_3, 0, sizeof(char_array_3));
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 3; j++) {
            char_array_3[j] = 0;
        }
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; (j < i + 1); j++) {
            encodedData.push_back(base64Characters[char_array_4[j]]);
        }

        while ((i++ < 3)) {
            encodedData.push_back('=');
        }
    }

    return encodedData;
}

std::vector<uint8_t> CHHashFileV::base64Decode(
    std::vector<uint8_t> charactersToDecode,
    std::string base64Characters) {

  int in_len = charactersToDecode.size();
  int i = 0;
  int j = 0;
  int in_ = 0;
  unsigned char char_array_4[4], char_array_3[3];
  std::vector<uint8_t> ret;

  while (in_len-- && ( charactersToDecode[in_] != '=') && is_base64(charactersToDecode[in_])) {
    char_array_4[i++] = charactersToDecode[in_]; in_++;
    if (i ==4) {
      for (i = 0; i <4; i++)
        char_array_4[i] = base64Characters.find(char_array_4[i]);

      char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
      char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

      for (i = 0; (i < 3); i++)
        ret.push_back(char_array_3[i]);
      i = 0;
    }
  }

  if (i) {
    for (j = i; j <4; j++)
      char_array_4[j] = 0;

    for (j = 0; j <4; j++)
      char_array_4[j] = base64Characters.find(char_array_4[j]);

    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

    for (j = 0; (j < i - 1); j++) ret.push_back(char_array_3[j]);
  }

  return ret;
}

// Default base64 functions

std::vector<uint8_t> CHHashFileV::base64Encode(
        std::vector<uint8_t> bytesToEncode) {
    // Use the default charset for base64
    return this->base64Encode(bytesToEncode, defaultBase64Charset);
}

std::vector<uint8_t> CHHashFileV::base64Decode(
    std::vector<uint8_t> charactersToDecode) {
    return this->base64Decode(charactersToDecode, defaultBase64Charset);
}
// Pulled from phpbb code, which appears to be GPLv2, so compatible.
// More or less a direct port.

std::vector<uint8_t> CHHashFileV::phpHash64Encode(
        std::vector<uint8_t> bytesToEncode,
        std::string base64Characters) {

    std::vector<uint8_t> output;
    uint32_t count = bytesToEncode.size();
    uint32_t i = 0, value;

    do {
        value = bytesToEncode[i++];
        output.push_back(base64Characters[value & 0x3f]);

        if (i < count) {
            value |= (((uint32_t) bytesToEncode[i]) << 8);
        }

        output.push_back(base64Characters[(value >> 6) & 0x3f]);

        if (i++ >= count) {
            break;
        }

        if (i < count) {
            value |= (((uint32_t) bytesToEncode[i]) << 16);
        }
        output.push_back(base64Characters[(value >> 12) & 0x3f]);

        if (i++ >= count) {
            break;
        }
        output.push_back(base64Characters[(value >> 18) & 0x3f]);
    } while (i < count);

    return output;
}

std::vector<uint8_t> CHHashFileV::phpHash64Decode(
    std::vector<uint8_t> charactersToDecode,
    std::string base64Characters) {
    
    std::vector<uint8_t> output;
    uint32_t charPos = 0;
    uint32_t data;
    
    // Lookup array to go from character to value more quickly
    uint8_t charLU[256];
    
    // Clear the lookup array & insert the values.
    memset(charLU, 0, 256);
    for (int i = 0; i < base64Characters.length(); i++) {
        charLU[base64Characters[i]] = i;
    }
    
    // Process the 4 character/3 byte chunks
    while ((charactersToDecode.size() - charPos) >= 4) {
        data = 0;
        data |= ((uint32_t)charLU[charactersToDecode[charPos++]]);
        data |= ((uint32_t)charLU[charactersToDecode[charPos++]] << 6);
        data |= ((uint32_t)charLU[charactersToDecode[charPos++]] << 12);
        data |= ((uint32_t)charLU[charactersToDecode[charPos++]] << 18);
        
        output.push_back(data & 0xff);
        output.push_back((data >> 8) & 0xff);
        output.push_back((data >> 16) & 0xff);
    }
    
    uint8_t bytesLeft = (charactersToDecode.size() - charPos);
    // There should be at least two bytes left of the input
    if ((charactersToDecode.size() - charPos) >= 2) {
        data = 0;
        data |= ((uint32_t)charLU[charactersToDecode[charPos++]]);
        data |= ((uint32_t)charLU[charactersToDecode[charPos++]] << 6);
        output.push_back(data & 0xff);
        if (bytesLeft >= 3) {
            data |= ((uint32_t)charLU[charactersToDecode[charPos++]] << 12);
            output.push_back((data >> 8) & 0xff);
        }
        // If there were 4 bytes left, it would have been covered above.
    } else if ((charactersToDecode.size() - charPos) == 1) {
        printf("Dangling character error.\n");
    }
    
    return output;
}

// Default function override

std::vector<uint8_t> CHHashFileV::phpHash64Encode(
        std::vector<uint8_t> bytesToEncode) {
    return this->phpHash64Encode(bytesToEncode, defaultPHPBBCharset);
}

std::vector<uint8_t> CHHashFileV::phpHash64Decode(
    std::vector<uint8_t> charactersToDecode) {
    return this->phpHash64Decode(charactersToDecode, defaultPHPBBCharset);
}

void CHHashFileV::testPHPPassHash() {
    static std::string PHPBBBase64 = "./0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    uint8_t testSixBytesArray[] = {0x37, 0xc8, 0x77, 0x6e, 0x6a, 0xb9};
    uint8_t testSixteenBytesArray[] = {0x7b, 0xfd, 0xf2, 0x90, 0x4e, 0x50, 0x3b,
        0x2b, 0xea, 0x22, 0x21, 0xb6, 0xab, 0x0b, 0xd5, 0x48};
    
    std::vector<uint8_t> testSixBytes;
    std::vector<uint8_t> testSixBytesEncoded;
    std::vector<uint8_t> testSixBytesDecoded;
    std::vector<uint8_t> testSixteenBytes;
    std::vector<uint8_t> testSixteenBytesEncoded;
    std::vector<uint8_t> testSixteenBytesDecoded;
    
    for (int i = 0; i < 6; i++) {
        testSixBytes.push_back(testSixBytesArray[i]);
    }
    
    testSixBytesEncoded = this->phpHash64Encode(testSixBytes, PHPBBBase64);
    
    printf("Encoded six bytes: ");
    for (int i = 0; i < testSixBytesEncoded.size(); i++) {
        printf("%c", (char)testSixBytesEncoded[i]);
    }
    printf("\n");
    printf("Expect  six bytes: rUwRidKi\n");
    
    // Decode & check
    testSixBytesDecoded = this->phpHash64Decode(testSixBytesEncoded, PHPBBBase64);

    printf("Decoded six bytes: ");
    for (int i = 0; i < testSixBytesDecoded.size(); i++) {
        printf("%02x", (uint8_t)testSixBytesDecoded[i]);
    }
    printf("\n");
    printf("Expect  six bytes: 37c8776e6ab9\n");
    
    

    for (int i = 0; i < 16; i++) {
        testSixteenBytes.push_back(testSixteenBytesArray[i]);
    }
    
    testSixteenBytesEncoded = this->phpHash64Encode(testSixteenBytes, PHPBBBase64);
    
    printf("Encoded sixteen bytes: ");
    for (int i = 0; i < testSixteenBytesEncoded.size(); i++) {
        printf("%c", (char)testSixteenBytesEncoded[i]);
    }
    printf("\n");
    printf("Expect  sixteen bytes: vpjwEu2IvgWuW2WhfiEp6/\n");

    // Decode & check
    testSixteenBytesDecoded = this->phpHash64Decode(testSixteenBytesEncoded, PHPBBBase64);

    printf("Decoded sixteen bytes: ");
    for (int i = 0; i < testSixteenBytesDecoded.size(); i++) {
        printf("%02x", (uint8_t)testSixteenBytesDecoded[i]);
    }
    printf("\n");
    printf("Expect  sixteen bytes: 7bfdf2904e503b2bea2221b6ab0bd548\n");
    
    // Stress test
    std::vector<uint8_t> rawData;
    std::vector<uint8_t> encodedData;
    std::vector<uint8_t> decodedData;
    
    printf("Starting stress test\n");
    for (int i = 0; i < 2048; i++) {
        // Add random byte
        rawData.push_back(rand() & 0xff);
        encodedData = this->phpHash64Encode(rawData, PHPBBBase64);
        decodedData = this->phpHash64Decode(encodedData, PHPBBBase64);
        
        if (decodedData != rawData) {
            printf("Data mismatch, length %d\n", i);
        }
    }
    printf("Stress test completed.\n");
}


std::string getHashFunctionByDefinedByte(uint8_t hashFunctionValue) {
    switch (hashFunctionValue) {
        case MFN_PASSWORD_NOT_FOUND:
            return std::string("NOT FOUND");
        case MFN_PASSWORD_SINGLE_MD5:
            return std::string("MD5");
        case MFN_PASSWORD_DOUBLE_MD5:
            return std::string("DMD5");
        case MFN_PASSWORD_TRIPLE_MD5:
            return std::string("TMD5");
        case MFN_PASSWORD_NTLM:
            return std::string("NTLM");
        case MFN_PASSWORD_SHA1:
            return std::string("SHA1");
        case MFN_PASSWORD_SHA1_OF_MD5:
            return std::string("SHA1_OF_MD5");
        case MFN_PASSWORD_MD5_OF_SHA1:
            return std::string("MD5_OF_SHA1");
        case MFN_PASSWORD_LM:
            return std::string("LM");
        case MFN_PASSWORD_SHA256:
            return std::string("SHA256");
        case MFN_PASSWORD_MD4:
            return std::string("MD4");
        default:
            return std::string("UNKNOWN");
    }
}
