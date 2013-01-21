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


#include <CH_Common/CHHashImplementation.h>
#include <stdio.h>

static char hexToAsciiLower[] = {'0', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
static char hexToAsciiUpper[] = {'0', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};

std::vector<std::vector<uint8_t> > CHHashImplementation::hashMultipleData(
        const std::vector<std::vector<uint8_t> > &rawMultipleData) {
    // Iterate over the list of multiple hashes and perform the single hash
    // function on each one.
    
    size_t i;
    std::vector<std::vector<uint8_t> > returnHashes;
    
    for (i = 0; i < rawMultipleData.size(); i++) {
        returnHashes.push_back(this->hashData(rawMultipleData[i]));
    }
    
    return returnHashes;
}

void CHHashImplementation::prepareMultipleHash(int passLength,
        std::vector<std::vector<uint8_t> > &rawMultipleHash) {
    // Iterate over the list of multiple hashes and perform the single hash
    // function on each one.
    
    uint64_t i;
    
    for (i = 0; i < rawMultipleHash.size(); i++) {
        this->prepareHash(passLength, rawMultipleHash[i]);
    }
}

std::vector<uint8_t> CHHashImplementation::hashDataAsciiVector(
    const std::vector<uint8_t> &rawData,
    uint8_t useUppercase) {
    
    std::vector<uint8_t> binaryHashOutput;
    std::vector<uint8_t> asciiHashOutput;

    // Hash the data
    binaryHashOutput = this->hashData(rawData);
    
    // Convert from hex to ascii
    for (int i = 0; i < binaryHashOutput.size(); i++) {
        uint8_t byteData = binaryHashOutput[i];
        if (useUppercase) {
            asciiHashOutput.push_back(hexToAsciiUpper[(byteData >> 4) & 0x0f]);
            asciiHashOutput.push_back(hexToAsciiUpper[byteData & 0x0f]);
        } else {
            asciiHashOutput.push_back(hexToAsciiLower[(byteData >> 4) & 0x0f]);
            asciiHashOutput.push_back(hexToAsciiLower[byteData & 0x0f]);
        }
    }
    return asciiHashOutput;
}

std::string CHHashImplementation::hashDataAsciiString(
    const std::vector<uint8_t> &rawData,
    uint8_t useUppercase) {
    
    std::vector<uint8_t> asciiHashOutput;

    asciiHashOutput = this->hashDataAsciiVector(rawData, useUppercase);
    
    return std::string(asciiHashOutput.begin(), asciiHashOutput.end());
}


std::vector<std::vector<uint8_t> > CHHashImplementation::hashMultipleDataAsciiVector(
        const std::vector<std::vector<uint8_t> > &rawMultipleData,
        uint8_t useUppercase) {
    // Iterate over the list of multiple hashes and perform the single hash
    // function on each one.
    
    size_t i;
    std::vector<std::vector<uint8_t> > returnHashes;
    
    for (i = 0; i < rawMultipleData.size(); i++) {
        returnHashes.push_back(this->hashDataAsciiVector(rawMultipleData[i],
                useUppercase));
    }
    
    return returnHashes;
}

std::vector<std::string> CHHashImplementation::hashMultipleDataAsciiString(
        const std::vector<std::vector<uint8_t> > &rawMultipleData,
        uint8_t useUppercase) {
    // Iterate over the list of multiple hashes and perform the single hash
    // function on each one.
    
    size_t i;
    std::vector<std::string> returnHashes;
    
    for (i = 0; i < rawMultipleData.size(); i++) {
        returnHashes.push_back(this->hashDataAsciiString(rawMultipleData[i],
                useUppercase));
    }
    
    return returnHashes;
}
