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

#include "GRT_Common/GRTTableHeaderVWeb.h"
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include "GRT_Common/GRTCharsetSingle.h"
#include <math.h>
#include "GRT_Common/GRTCrackDisplay.h"

#include <string>
#include <vector>
#include <curl/curl.h>

#define UNIT_TEST 0

std::string GRTTableHeaderErrorMessage;

// Handle a write from the "is table valid" bit.
size_t valid_table_write(void *buffer, size_t size, size_t nmemb, void *userp) {

    char *validTable = (char *)userp;
    char *bufferArray = (char *)buffer;
    if ((size * nmemb) == 1) {
        // String is NOT zero terminated - check the first character.
        if (bufferArray[0] == '1') {
            *validTable = 1;
        }
        return 1;
    } else {
        //printf("Size received: %d\n", size * nmemb);
        //printf("valid_table_write error: %s\n", (char *)buffer);
        // An error has happened.  The text will be useful.
        GRTTableHeaderErrorMessage = std::string((char *)buffer, (size * nmemb));
        return 0;
    }
    return 0;
}

// Handle a read pf the table header
size_t table_header_write(void *buffer, size_t size, size_t nmemb, void *userp) {

    // Get us a vector pointer.
    std::vector<uint8_t> *headerBuffer = (std::vector<uint8_t> *)userp;
    uint8_t *bufferPointer = (uint8_t *)buffer;

    for (int i = 0; i < (size * nmemb); i++) {
        headerBuffer->push_back(bufferPointer[i]);
    }
    //printf("Size received: %d\n", size * nmemb);

    return (size * nmemb);
}

// Handle a read pf the table header
size_t table_filenames_write(void *buffer, size_t size, size_t nmemb, void *userp) {

    // Get us a vector pointer.
    std::vector<uint8_t> *filenamesBuffer = (std::vector<uint8_t> *)userp;
    uint8_t *bufferPointer = (uint8_t *)buffer;

    for (int i = 0; i < (size * nmemb); i++) {
        filenamesBuffer->push_back(bufferPointer[i]);
    }
    //printf("Size received: %d\n", size * nmemb);

    return (size * nmemb);
}

GRTTableHeaderVWeb::GRTTableHeaderVWeb() {
    // Clear the table header.
    memset(&this->Table_Header, 0, sizeof(this->Table_Header));

    this->tableValid = 0;
}

// Checks to ensure a table exists.
char GRTTableHeaderVWeb::isValidTable(const char *filename, int hashVersion) {
    CURL *curl;
    CURLcode res;

    char buffer[256];
    std::string postString;

    memset(buffer, 0, 256);
    this->tableValid = 0;

    postString = "isValidTable=";
    postString += filename;

    // If the hash version is requested (not -1), check it.
    if (hashVersion >= 0) {
        // add hash version request
        sprintf(buffer, "%d", hashVersion);
        postString += "&hashVersion=";
        postString += buffer;
    }

    //printf("Submission string: %s\n", postString.c_str());
    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, this->tableURL.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postString.c_str());
        // Pass a pointer to our tableValid variable for the callback.
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &this->tableValid);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, valid_table_write);

        // If we have a username, set username/password/authentication.
        if (this->tableUsername.length()) {
            curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
            curl_easy_setopt(curl, CURLOPT_USERNAME, this->tableUsername.c_str());
            curl_easy_setopt(curl, CURLOPT_PASSWORD, this->tablePassword.c_str());
        }

        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            // Error: Something is wrong.
            printf("\n\n========WebTables Error========\n%s\n\n", GRTTableHeaderErrorMessage.c_str());
            curl_easy_cleanup(curl);
            exit(1);
            return 0;

        }
        /* always cleanup */
        curl_easy_cleanup(curl);
    }

    // We should have the result now - return it.
    return this->tableValid;
}

char GRTTableHeaderVWeb::readTableHeader(const char *filename){

    CURL *curl;
    CURLcode res;

    std::string postString;

    std::vector<uint8_t> headerBuffer;

    headerBuffer.reserve(8192);

    postString = "readTableHeader=";
    postString += filename;

    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, this->tableURL.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postString.c_str());
        // Pass a pointer to our tableValid variable for the callback.
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &headerBuffer);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, table_header_write);

        // If we have a username, set username/password/authentication.
        if (this->tableUsername.length()) {
            curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
            curl_easy_setopt(curl, CURLOPT_USERNAME, this->tableUsername.c_str());
            curl_easy_setopt(curl, CURLOPT_PASSWORD, this->tablePassword.c_str());
        }


        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            // Error: Something is wrong.
            printf("curl error in readTableHeader: %s\n", curl_easy_strerror(res));
            curl_easy_cleanup(curl);
            return 0;
        }
        /* always cleanup */
        curl_easy_cleanup(curl);
    }

    if (headerBuffer.size() != 8192) {
        if (this->Display) {
            this->Display->endCursesMode();
        }
        printf("\n\n========WebTables Error========\n");
        headerBuffer.push_back(0);
        printf("%s\n", (char *)&headerBuffer[0]);
        return 0;
    } else {
        memcpy(&this->Table_Header, &headerBuffer[0], 8192);
    }


    // We should have the result now - return it.
    return 1;
};

void GRTTableHeaderVWeb::printTableHeader(){
    // Print out the table metainfo.
    printf("\n");
    printf("Table version:   %d\n", this->Table_Header.TableVersion);
    printf("Hash:            %s\n", this->Table_Header.HashName);
    printf("Password length: %d\n", this->Table_Header.PasswordLength);
    printf("Table index:     %d\n", this->Table_Header.TableIndex);
    printf("Chain length:    %d\n", this->Table_Header.ChainLength);
    printf("Num chains:      %ld\n", this->Table_Header.NumberChains);
    printf("Perfect table:   ");
    if (this->Table_Header.IsPerfect) {
        printf("Yes\n");
    } else {
        printf("No\n");
    }
    printf("Charset length:  %d\n", this->Table_Header.CharsetLength[0]);
    printf("Charset:         ");
    for (int i = 0; i < this->Table_Header.CharsetLength[0]; i++) {
        printf("%c", this->Table_Header.Charset[0][i]);
        // Add a newline at sane points.
        if ((i % 50 == 0) && (i)) {
            printf("\n                 ");
        }
    }
    printf("\nBits of hash:    %d\n", this->Table_Header.BitsInHash);
    printf("Bits of pass:    %d\n", this->Table_Header.BitsInPassword);

    if (this->Table_Header.TableVersion == 3) {
        printf("Random Seed:    %lu\n", this->Table_Header.randomSeedValue);
        printf("Chain start offset:    %lu\n", this->Table_Header.chainStartOffset);
    }

    printf("\n\n");
};

char* GRTTableHeaderVWeb::getHashName(){
    char *HashNameReturn;

    HashNameReturn = new char(16);
    strcpy(HashNameReturn, (const char *)&this->Table_Header.HashName);
    return HashNameReturn;
};

char* GRTTableHeaderVWeb::getCharsetLengths(){
    char *ReturnCharsetLengths;
    int i;

    ReturnCharsetLengths = new char[16];

    for (i = 0; i < 16; i++) {
        ReturnCharsetLengths[i] = this->Table_Header.CharsetLength[i];
    }
    return ReturnCharsetLengths;
};

char** GRTTableHeaderVWeb::getCharset(){
    int i, j;

    char **ReturnCharsetArray = new char*[16];
    for (i = 0; i < 16; i++)
        ReturnCharsetArray[i] = new char[256];

    for (i = 0; i < 16; i++) {
        for (j = 0; j < 256; j++) {
            ReturnCharsetArray[i][j] = this->Table_Header.Charset[i][j];
        }
    }

    return ReturnCharsetArray;
};


char* GRTTableHeaderVWeb::getComments(){return NULL;};

std::vector<std::string>  GRTTableHeaderVWeb::getHashesFromServerByType(int hashType) {
    //printf("GRTTableHeaderVWeb::getHashesFromServerByType(%d)\n", hashType);

    CURL *curl;
    CURLcode res;

    char buffer[256];
    std::string postString;

    std::vector<uint8_t> filenamesBuffer;

    std::vector<std::string> returnFilenames;
    std::string filename;

    memset(buffer, 0, 256);
    this->tableValid = 0;

    // Set the hash ID to get tables of
    postString = "getTableListByHashId=";
    sprintf(buffer, "%d", hashType);
    postString += buffer;

    //printf("Submission string: %s\n", postString.c_str());
    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, this->tableURL.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postString.c_str());
        // Pass a pointer to our tableValid variable for the callback.
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &filenamesBuffer);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, table_filenames_write);

        // If we have a username, set username/password/authentication.
        if (this->tableUsername.length()) {
            curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
            curl_easy_setopt(curl, CURLOPT_USERNAME, this->tableUsername.c_str());
            curl_easy_setopt(curl, CURLOPT_PASSWORD, this->tablePassword.c_str());
        }
        
        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            // Error: Something is wrong.
            printf("curl error: %s\n", curl_easy_strerror(res));
            //exit(1);
            curl_easy_cleanup(curl);
            return returnFilenames;

        }
        /* always cleanup */
        curl_easy_cleanup(curl);
    }

    filenamesBuffer.push_back(0);
    //printf("Got buffer back: %s\n", (char *)&filenamesBuffer[0]);


    for (int i = 0; i < filenamesBuffer.size(); i++) {
        if (filenamesBuffer.at(i) == '\n') {
            //printf("Pushing back %s\n", filename.c_str());
            returnFilenames.push_back(filename);
            filename.clear();
            continue;
        }
        filename += filenamesBuffer.at(i);
    }
    return returnFilenames;

}

std::vector<uint8_t> GRTTableHeaderVWeb::getHeaderString() {
    std::vector<uint8_t> returnVector;

    // Set the size to the header length before we copy it in.
    returnVector.resize(sizeof(this->Table_Header));
    memcpy(&returnVector[0], &this->Table_Header, sizeof(this->Table_Header));

    return returnVector;
}


#if UNIT_TEST

#include <string.h>
int main() {
    GRTTableHeaderVWeb TableHeader;

    printf("Test!\n");

    if (TableHeader.isValidTable("MD5-len8-idx0-chr95-cl1000-sd2498211041-0-v2.part", 0)) {
        printf("Table is valid!\n");
    } else {
        printf("Table is NOT valid!\n");
    }

    TableHeader.readTableHeader("NTLM-len8-idx0-chr10-cl5000-sd1482662457-0-v2.part");
    

    TableHeader.printTableHeader();
}
#endif



