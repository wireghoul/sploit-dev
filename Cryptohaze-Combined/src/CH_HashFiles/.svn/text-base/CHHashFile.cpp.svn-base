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

#include "CH_HashFiles/CHHashFile.h"
#include "MFN_Common/MFNDebugging.h"

#include <vector>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <fstream>

static inline bool is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

// Default, bog standard base64 charsets for different functions.
static std::string defaultBase64Charset =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static std::string defaultPHPBBCharset =
        "./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

std::vector<uint8_t> CHHashFile::convertAsciiToBinary(std::string asciiHex) {
    trace_printf("CHHashFile::convertAsciiToBinary(std::string)\n");
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

std::vector<uint8_t> CHHashFile::convertAsciiToBinary(std::vector<char> asciiHex) {
    trace_printf("CHHashFile::convertAsciiToBinary(std::vector<char>)\n");
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

std::vector<uint8_t> CHHashFile::base64Encode(
        std::vector<uint8_t> bytesToEncode,
        std::string base64Characters) {
    trace_printf("CHHashFile::base64Encode()\n");

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
            memset(char_array_3, 0, sizeof (char_array_3));
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

std::vector<uint8_t> CHHashFile::base64Decode(
        std::vector<uint8_t> charactersToDecode,
        std::string base64Characters) {
    trace_printf("CHHashFile::base64Decode()\n");

    int in_len = charactersToDecode.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::vector<uint8_t> ret;

    while (in_len-- && (charactersToDecode[in_] != '=') && is_base64(charactersToDecode[in_])) {
        char_array_4[i++] = charactersToDecode[in_];
        in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++)
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
        for (j = i; j < 4; j++)
            char_array_4[j] = 0;

        for (j = 0; j < 4; j++)
            char_array_4[j] = base64Characters.find(char_array_4[j]);

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; (j < i - 1); j++) ret.push_back(char_array_3[j]);
    }

    return ret;
}

std::vector<uint8_t> CHHashFile::base64Encode(
        std::vector<uint8_t> bytesToEncode) {
    // Use the default charset for base64
    return this->base64Encode(bytesToEncode, defaultBase64Charset);
}

std::vector<uint8_t> CHHashFile::base64Decode(
        std::vector<uint8_t> charactersToDecode) {
    return this->base64Decode(charactersToDecode, defaultBase64Charset);
}

// Pulled from phpbb code, which appears to be GPLv2, so compatible.
// More or less a direct port.

std::vector<uint8_t> CHHashFile::phpHash64Encode(
        std::vector<uint8_t> bytesToEncode,
        std::string base64Characters) {
    trace_printf("CHHashFile::phpHash64Encode()\n");

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

std::vector<uint8_t> CHHashFile::phpHash64Decode(
        std::vector<uint8_t> charactersToDecode,
        std::string base64Characters) {
    trace_printf("CHHashFile::phpHash64Decode()\n");

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
        data |= ((uint32_t) charLU[charactersToDecode[charPos++]]);
        data |= ((uint32_t) charLU[charactersToDecode[charPos++]] << 6);
        data |= ((uint32_t) charLU[charactersToDecode[charPos++]] << 12);
        data |= ((uint32_t) charLU[charactersToDecode[charPos++]] << 18);

        output.push_back(data & 0xff);
        output.push_back((data >> 8) & 0xff);
        output.push_back((data >> 16) & 0xff);
    }

    uint8_t bytesLeft = (charactersToDecode.size() - charPos);
    // There should be at least two bytes left of the input
    if ((charactersToDecode.size() - charPos) >= 2) {
        data = 0;
        data |= ((uint32_t) charLU[charactersToDecode[charPos++]]);
        data |= ((uint32_t) charLU[charactersToDecode[charPos++]] << 6);
        output.push_back(data & 0xff);
        if (bytesLeft >= 3) {
            data |= ((uint32_t) charLU[charactersToDecode[charPos++]] << 12);
            output.push_back((data >> 8) & 0xff);
        }
        // If there were 4 bytes left, it would have been covered above.
    } else if ((charactersToDecode.size() - charPos) == 1) {
        printf("Dangling character error.\n");
    }

    return output;
}

// Default function override

std::vector<uint8_t> CHHashFile::phpHash64Encode(
        std::vector<uint8_t> bytesToEncode) {
    return this->phpHash64Encode(bytesToEncode, defaultPHPBBCharset);
}

std::vector<uint8_t> CHHashFile::phpHash64Decode(
        std::vector<uint8_t> charactersToDecode) {
    return this->phpHash64Decode(charactersToDecode, defaultPHPBBCharset);
}

int CHHashFile::openHashFile(std::string filename) {
    trace_printf("CHHashFile::openHashFile()\n");
    std::ifstream hashFile;
    std::string fileLine;
    size_t currentLineNumber = 0;

    // Don't strip spaces - some salts use them.
    std::string whitespaces("\t\f\v\n\r");
    size_t found;

    this->HashFileMutex.lock();

    hashFile.open(filename.c_str(), std::ios_base::in);
    if (!hashFile.good()) {
        printf("ERROR: Cannot open hashfile %s\n", filename.c_str());
        exit(1);
    }

    while (std::getline(hashFile, fileLine)) {
        // Always increment the line number.  Most editors report the first line
        // as line 1, so this will be correct.
        currentLineNumber++;

        // Look for and trim trailing newlines/etc.
        found = fileLine.find_last_not_of(whitespaces);
        if (found != std::string::npos) {
            fileLine.erase(found + 1);
        } else {
            fileLine.clear();
        }

        // If the line length is 0, continue - blank line that we can ignore.
        if (fileLine.length() == 0) {
            continue;
        }

        // Let the leaf class parse the line.
        this->parseFileLine(fileLine, currentLineNumber);
    }

    // Done reading from the hash file.  Close it and perform whatever needs
    // to happen next.
    hashFile.close();
    this->performPostLoadOperations();

    // If NO hashes are loaded, something is probably very wrong.
    if (this->TotalHashes == 0) {
        printf("No hashes loaded!\n");
        exit(1);
    }

    this->HashFileMutex.unlock();
    return 1;
}

void CHHashFile::exportHashListToRemoteSystem(std::string &exportData) {
    trace_printf("CHHashFile::exportHashListToRemoteSystem()\n");
    this->HashFileMutex.lock();

    // If the cache is not valid, create it.
    if (!this->protobufExportCachesValid) {
        this->createHashListExportProtobuf();
        this->createUniqueSaltsExportProtobuf();
        this->protobufExportCachesValid = 1;
    }

    // Copy the valid data.
    exportData = this->hashExportProtobufCache;

    this->HashFileMutex.unlock();
}

void CHHashFile::exportUniqueSaltsToRemoteSystem(std::string &exportData) {
    trace_printf("CHHashFile::exportUniqueSaltsToRemoteSystem()\n");
    this->HashFileMutex.lock();

    // If the cache is not valid, create it.
    if (!this->protobufExportCachesValid) {
        this->createHashListExportProtobuf();
        this->createUniqueSaltsExportProtobuf();
        this->protobufExportCachesValid = 1;
    }

    // Copy the valid data.
    exportData = this->saltExportProtobufCache;

    this->HashFileMutex.unlock();
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
 