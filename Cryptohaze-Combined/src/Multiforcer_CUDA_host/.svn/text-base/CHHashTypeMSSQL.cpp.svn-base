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

#include "Multiforcer_CUDA_host/CHHashTypeMSSQL.h"
#include "Multiforcer_Common/CHCommon.h"

#include "Multiforcer_Common/CHDisplay.h"
#include "Multiforcer_CUDA_host/CHCommandLineData.h"
#include "CH_Common/CHHashFileMSSQL.h"
#include "CH_Common/CHCharset.h"


extern "C" {
    void *CHHashTypeGPUThread(void *);
}

// Call the constructor of CHHashTypePlain with len 20
CHHashTypeMSSQL::CHHashTypeMSSQL() : CHHashTypePlain(20) {
    
}


void CHHashTypeMSSQL::copyDataToConstant(char *hostCharset, int charsetLength,
        unsigned char *hostCharsetLengths, unsigned char *hostSharedBitmap, int threadId) {
    copyMSSQLDataToConstant(hostCharset, charsetLength, hostCharsetLengths,
            hostSharedBitmap, threadId, this->SaltList);
}

void CHHashTypeMSSQL::Launch_CUDA_Kernel(int passlength, uint64_t charsetLength, int numberOfPasswords, unsigned char *DEVICE_Passwords,
            unsigned char *DEVICE_Success, struct start_positions *DEVICE_Start_Positions,
        uint64_t per_step, uint64_t threads, uint64_t blocks, unsigned char *DEVICE_Hashes,
        unsigned char *DEVICE_Bitmap) {
    // Pass through to the device file
    Launch_CUDA_MSSQL_Kernel(passlength, charsetLength, numberOfPasswords,
                DEVICE_Passwords, DEVICE_Success,
                DEVICE_Start_Positions, per_step,
                threads, blocks, DEVICE_Hashes,
                DEVICE_Bitmap);
}

// Do the main work of this type.
void CHHashTypeMSSQL::crackPasswordLength(int passwordLength) {
    int i;
    this->Display->setPasswordLen(passwordLength);
    this->passwordLength = passwordLength;

    // Do the global work
    this->HashList = this->HashFile->ExportUncrackedHashList();

    // Need this added.
    this->SaltList = this->HashFile->GetSaltList();
    this->NumberOfHashes = this->HashFile->GetUncrackedHashCount();
    this->hostConstantCharset = this->Charset->getCharset();
    this->hostConstantCharsetLength = this->Charset->getCharsetNumberElements();

    this->Display->setTotalHashes(this->NumberOfHashes);

    for (i = 0; i < passwordLength; i++) {
        this->hostCharsetLengths[i] = this->Charset->getCharsetLength(i);
    }


    this->createConstantBitmap8kb();
    // If we need the big lookup table, create it.
    if (this->CommandLineData->GetUseLookupTable()) {
        this->createGlobalBitmap512MB();
    }

#if USE_BOOST_THREADS
    memset(this->ThreadObjects, 0, MAX_SUPPORTED_THREADS * sizeof(boost::thread *));
#else
    memset(this->ThreadIds, 0, MAX_SUPPORTED_THREADS * sizeof(pthread_t));
#endif
    
    // All global work done.  Onto the per-thread work.
    sprintf(this->statusBuffer, "Creating %d threads\n", this->ActiveThreadCount);
    this->Display->addStatusLine(this->statusBuffer);
    // Enter all the threads
    for(i = 0; i < this->ActiveThreadCount; i++) {
#if USE_BOOST_THREADS
        this->ThreadObjects[i] = new boost::thread(&CHHashTypeGPUThread, &this->ThreadData[i]);
#else
        pthread_create(&this->ThreadIds[i], NULL, CHHashTypeGPUThread, &this->ThreadData[i] );
#endif
    }
    // Wait for them to come back.
    for(i = 0; i < this->ActiveThreadCount; i++) {
#if USE_BOOST_THREADS
        this->ThreadObjects[i]->join();
#else
        pthread_join(this->ThreadIds[i], NULL);
#endif
    }
    sprintf(this->statusBuffer, "Threads joined\n");
    this->Display->addStatusLine(this->statusBuffer);
}

// Overriding, we will only have a MSSQL hash file type.
void CHHashTypeMSSQL::setHashFile(CHHashFileTypes *NewHashFile) {
    this->HashFile = (CHHashFileMSSQL*)NewHashFile;
}

/* outputFoundHashes takes the various arrays, looks for newly found passwords, prints them, and outputs them to the output file if needed.
 * It returns the number of passwords it found that were new.
 */
int CHHashTypeMSSQL::outputFoundHashes(struct threadRunData *data) {
  int i, j;
  int passwordsFound = 0;
    for (i = 0; i < this->NumberOfHashes; i++) {
        if (this->HOST_Success[data->threadID][i] && !this->HOST_Success_Reported[data->threadID][i]) {
            this->HashFile->ReportFoundPassword(&this->HashList[i * this->hashLengthBytes],
                    &this->HOST_Passwords[data->threadID][MAX_PASSWORD_LEN * i]);
            passwordsFound++;
            this->HOST_Success_Reported[data->threadID][i] = 1;

            for (j = 0; j < this->passwordLength; j++) {
                this->statusBuffer[j] = this->HOST_Passwords[data->threadID][MAX_PASSWORD_LEN * i + j];
            }
            this->statusBuffer[j] = 0;
            this->Display->addCrackedPassword(this->statusBuffer);
        }
    }
  this->Display->addCrackedHashes(passwordsFound);
  return passwordsFound;
}