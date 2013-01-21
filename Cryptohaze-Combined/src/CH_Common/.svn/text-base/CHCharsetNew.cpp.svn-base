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

#include "CH_Common/CHCharsetNew.h"

#include <stdlib.h>
#include <string.h>
#include "Multiforcer_Common/CHCommon.h"
#include <algorithm>

extern struct global_commands global_interface;

uint8_t CHCharsetNew::readCharsetFromFile(std::string charsetFilename){
    std::ifstream charsetFile;
    std::string fileLine;
    int charsetLine = 0, charsetPosition = 0;
    std::vector<uint8_t> charsetLineData;

    this->Charset.clear();
    
    charsetFile.open(charsetFilename.c_str(), std::ios_base::in);
    
    if (!charsetFile.good())
    {
        
        std::cerr << "ERROR: Cannot open charset file " << charsetFilename <<"\n";
        exit(1);
    }
    
    //Read file, if valid. Error if the charset row is too long.
    while (charsetFile.good())
    {
        charsetPosition = 0;
        std::getline(charsetFile, fileLine);

        //std::cout << fileLine << std::endl;

        //printf("Charset line length: %d\n", fileLine.length());
        
        if (fileLine.length() > MAX_CHARSET_LENGTH)
        {
            std::cerr << "Charset in row "<< this->getCharsetNumberElements() <<"too long!\n";
            exit(1);
        }

        // If the line length is non-zero...
        if (fileLine.length()) {
            charsetLineData.clear();
            while(charsetPosition < fileLine.length() && fileLine[charsetPosition] != '\0')
            {
                charsetLineData.push_back(fileLine[charsetPosition]);
                charsetPosition++;
            }
            this->Charset.push_back(charsetLineData);
            charsetLine++;
        }
    }
    
    // Sort the charsets for better performance.
    for (int i = 0; i < this->Charset.size(); i++) {
        std::sort(this->Charset[i].begin(), this->Charset[i].end());
        this->Charset[i].erase(
                std::unique(this->Charset[i].begin(), this->Charset[i].end()),
                this->Charset[i].end() ); 
    }
    
    charsetFile.close();
    return 1;
}
void CHCharsetNew::clearCharset()
{
    this->Charset.erase(this->Charset.begin(), this->Charset.end());
    return;
}

uint16_t CHCharsetNew::getCharsetLength(uint16_t position)
{
    // Position 1 is valid if length = 2, position 2 is not.
    if (position >= this->Charset.size()) {
        return 0;
    }
    return (uint16_t)this->Charset[position].size();
}

uint64_t CHCharsetNew::getPasswordSpaceSize(uint16_t passwordLength){
    uint64_t passwordSpaceSize = 1;
    uint64_t overflowDetection = 1;
    int i;

    // If the length is one, return as a single charset number.
    if (this->getCharsetNumberElements() == 1) {
        for (i=0; i < passwordLength; i++) {
            passwordSpaceSize *= this->getCharsetLength(0);

            // If overflow, fail.
            if (passwordSpaceSize < overflowDetection) {
                sprintf(global_interface.exit_message, "Password space > 2^64 not supported!\n");
                global_interface.exit = 1;
                return 0;
            }
            overflowDetection = passwordSpaceSize;
        }
    } else if (this->getCharsetNumberElements() < passwordLength) {
            sprintf(global_interface.exit_message, "Charset does not extend to password length %d!\n", passwordLength );
            global_interface.exit = 1;
            return 0;
    } else {
        // Find the space, and check for overflows.  We don't support greater
        // than 2^64 right now.
        for (i=0; i < passwordLength; i++) {
            passwordSpaceSize *= this->getCharsetLength(i);
            // If overflow, fail.
            if (passwordSpaceSize < overflowDetection) {
                sprintf(global_interface.exit_message, "Password space > 2^64 not supported!\n");
                global_interface.exit = 1;
                return 0;
            }
            overflowDetection = passwordSpaceSize;
        }
    }
    
    return passwordSpaceSize;
}

uint8_t *CHCharsetNew::exportLegacyCharset()
{
    int i;
    int position = 0;
    uint8_t* legacyCharset = new uint8_t[MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN];
    memset(legacyCharset, '\0', MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN);
    
    for(i=0; i < this->getCharsetNumberElements(); i++)
    {
        
        memcpy(legacyCharset, &this->Charset.at(i), this->getCharsetLength(i));
        position += this->getCharsetLength(i);
        
    }
    return legacyCharset;
    
}
void CHCharsetNew::loadRemoteCharsetIntoCharset(uint8_t* remoteCharset)
{

    int i=0;
    while(remoteCharset[i] != (uint8_t)'\0')
    {
        this->Charset.push_back( std::vector<uint8_t>(remoteCharset[i]) );
        i+=1;
    }
    return;

}
void CHCharsetNew::addCharacterAtPosition(uint16_t charsetPosition, uint8_t characterValue)
{
    int i;
    uint16_t numCharsetElements;
    
    numCharsetElements = this->getCharsetNumberElements();
    
    if ( numCharsetElements < charsetPosition)
    {
        for(i=0; i < charsetPosition - numCharsetElements; i++)
        {
            this->Charset.push_back(std::vector<uint8_t>('\0'));
        }
    }

    this->Charset[charsetPosition].push_back(characterValue);

    return;
}


void CHCharsetNew::ExportCharsetToRemoteSystem(std::string * exportData) {
    this->Protobuf.Clear();
    std::string charsetBuffer;
    
    std::vector<std::vector<uint8_t> >::iterator i;
    for(i=this->Charset.begin(); i<this->Charset.end();i++)
    {
        charsetBuffer = std::string(i->begin(), i->end());
        this->Protobuf.add_charset_position_data(charsetBuffer);
    }
    this->Protobuf.set_charset_size((uint32_t)this->Charset.size());
    
    //Danger: Please be sure to have some storage allocated to this pointer.
    //I shouldn't have to say this, but I will anyway.
    this->Protobuf.SerializeToString(exportData);
}

void CHCharsetNew::ImportCharsetFromRemoteSystem(std::string &remoteData) {
    this->Protobuf.Clear();
    this->Charset.clear();

    std::string charsetBuffer;
    std::vector<uint8_t> charsetElementVector;
    
    //Unpack protobuf
    this->Protobuf.ParseFromString(remoteData);
    
    //Extract the data from the protobuf
    for(int i=0;i<this->Protobuf.charset_position_data_size();i++)
    {
        charsetBuffer = this->Protobuf.charset_position_data(i);
        charsetElementVector.clear();
        for (int j = 0; j < charsetBuffer.length(); j++) {
            charsetElementVector.push_back(charsetBuffer[j]);
        }
        this->Charset.push_back(charsetElementVector);
    }

    this->Protobuf.Clear();
}
