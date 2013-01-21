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

// Main CUDA generate code.

#include "GRT_CUDA_host/GRTGenCommandLineData.h"
#include "GRT_Common/GRTHashes.h"
#include "GRT_Common/GRTTableHeaderV1.h"
#include "GRT_Common/GRTCharsetSingle.h"
#include "GRT_CUDA_host/GRTGenerateTableMD5.h"
#include "GRT_CUDA_host/GRTGenerateTableNTLM.h"
#include "GRT_CUDA_host/GRTGenerateTableSHA1.h"
#include "GRT_CUDA_host/GRTGenerateTableSHA256.h"
#include "CH_Common/CHRandom.h"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <curl/curl.h>

#ifdef _WIN32
	#include <Windows.h>
	#define GRTSleep(x) Sleep(x*1000)
#else
	#define GRTSleep(x) sleep(x)
#endif



// Silence output if true.
char silent = 0;

const char programTitle[] = "Cryptohaze GRTGen 1.00";

extern size_t table_upload_write(void *buffer, size_t size, size_t nmemb, void *userp);

int getCurrentHashId(GRTGenCommandLineData *TableParameters) {
    CURL *curl;
    CURLcode res;
    std::vector <uint8_t> returnBuffer;
    long http_code = 0;
    
    curl = curl_easy_init();
    if (curl) {
        char postString[] = "getAlgorithmName=get";
        // Add the elements to the form.
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postString);
        curl_easy_setopt(curl, CURLOPT_URL, TableParameters->getWebGenURL().c_str());
        // If we have a username, set username/password/authentication.
        if (TableParameters->getWebUsername().length()) {
            curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
            curl_easy_setopt(curl, CURLOPT_USERNAME, TableParameters->getWebUsername().c_str());
            curl_easy_setopt(curl, CURLOPT_PASSWORD, TableParameters->getWebPassword().c_str());
        }
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &returnBuffer);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, table_upload_write);

        res = curl_easy_perform(curl);
        curl_easy_getinfo (curl, CURLINFO_RESPONSE_CODE, &http_code);
        
        // Check for curl success.
        if (res != CURLE_OK) {
            // Error: Something is wrong.
            printf("========WebGen Error========\n");
            printf("curl error: %s\n", curl_easy_strerror(res));
            exit(1);
        }
        
        // Check for http success (we want a 200).
        if (http_code != 200) {
            printf("========WebGen Error========\n");
            printf("http error: %d\n", http_code);
            exit(1);
        }
        
        /*
        printf("\n\n");
        printf("Upload status: ");
        for (int charpos = 0; charpos < returnBuffer.size(); charpos++) {
            printf("%c", returnBuffer[charpos]);
        }
        */
        
        printf("\n\n");
        /* always cleanup */
        curl_easy_cleanup(curl);
    }
    returnBuffer.push_back(0);
    return atoi((char *)&returnBuffer[0]);}

int main(int argc, char *argv[]) {
    int deviceCount;

    GRTGenCommandLineData GenerateParams;
    GRTCharsetSingle Charset;
    GRTGenerateTable *GenTable;
    CHRandom RandomGenerator;

    printf("\n%s\n\n", programTitle);


    // Check to see if this is even a CUDA capable system.
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
      printf("This program requires a CUDA-capable video card.\nNo cards found.  Sorry.  Exiting.\n");
      exit(1);
    }

    //GenTable = new GRTGenerateTableMD5();

    // Parse the command line.  If this returns, things went well.
    GenerateParams.setRandomGenerator(&RandomGenerator);
    GenerateParams.ParseCommandLine(argc, argv);

    if (!GenerateParams.getUseWebGenerate()) {
        Charset.getCharsetFromFile(GenerateParams.GetCharsetFileName());
    }

    // We will loop back to here if needed.
    generateLoop:
    
    // If WebGen is being used, get the hash type and specify a reasonably
    // unliminted number of tables.
    if (GenerateParams.getUseWebGenerate()) {
        GenerateParams.setHashType(getCurrentHashId(&GenerateParams));
        GenerateParams.setNumberTables(0xffffffff);
    }

    switch(GenerateParams.getHashType()) {
        case 0:
            GenTable = new GRTGenerateTableNTLM();
            break;
        case 1:
            GenTable = new GRTGenerateTableMD5();
            break;
        case 3:
            GenTable = new GRTGenerateTableSHA1();
            break;
        case 4:
            GenTable = new GRTGenerateTableSHA256();
            break;
        default:
            printf("This hash type is not supported yet!\n");
            exit(1);
            break;
    }

    GenerateParams.PrintTableData();
    
    GenTable->setRandomGenerator(&RandomGenerator);
    GenTable->setGRTGenCommandLineData(&GenerateParams);
    GenTable->setGRTCharsetSingle(&Charset);
    GenTable->createTables();

    // Here for two reasons: Done generating tables, or a WebGen error.
    delete GenTable;
    
    // If WebTables is in use, loop back and try again.
    if (GenerateParams.getUseWebGenerate()) {
        GRTSleep(5);
        goto generateLoop;
    }
}
