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

#include "Multiforcer_CUDA_host/CHHashTypeSalted.h"
#include "Multiforcer_Common/CHDisplay.h"
#include "Multiforcer_CUDA_host/CHCommandLineData.h"
#include "CH_Common/CHHashFileMSSQL.h"
#include "CH_Common/CHCharset.h"
#include "CH_Common/CHHashFileSalted32.h"
#include "Multiforcer_Common/CHCommon.h"

extern struct global_commands global_interface;


extern "C" {
    void *CHHashTypeGPUThread(void *);
}

// Call the constructor of CHHashTypePlain with len 20
CHHashTypeSalted::CHHashTypeSalted(int newHashLength, int newSaltLength)
    : CHHashTypePlain(newHashLength) {
    this->SaltLength = newSaltLength;
}



// Overriding, we will only have a salted hash file type.
void CHHashTypeSalted::setHashFile(CHHashFileTypes *NewHashFile) {
    this->HashFile = (CHHashFileSalted32*)NewHashFile;
}


/* outputFoundHashes takes the various arrays, looks for newly found passwords, prints them, and outputs them to the output file if needed.
 * It returns the number of passwords it found that were new.
 */
// Needed because we have the new type hashfile
int CHHashTypeSalted::outputFoundHashes(struct threadRunData *data) {
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

  // Check to see if we should exit (as all hashes are found).
  if (this->HashFile->GetUncrackedHashCount() == 0) {
      global_interface.exit = 1;
  }

  return passwordsFound;
}


// Do the main work of this type.
void CHHashTypeSalted::crackPasswordLength(int passwordLength) {
    int i;

    this->threadRendezvous = 0;

    this->Display->setPasswordLen(passwordLength);
    this->passwordLength = passwordLength;

    // Do the global work
    this->HashList = this->HashFile->ExportUncrackedHashList();

    // Need this added.
    this->SaltList = this->HashFile->GetSaltList();
    this->SaltLengths = this->HashFile->GetSaltLengths();

    this->NumberOfHashes = this->HashFile->GetUncrackedHashCount();
    this->hostConstantCharset = this->Charset->getCharset();
    this->hostConstantCharsetLength = this->Charset->getCharsetNumberElements();

    this->Display->setTotalHashes(this->NumberOfHashes);

    for (i = 0; i < passwordLength; i++) {
        this->hostCharsetLengths[i] = this->Charset->getCharsetLength(i);
    }


    this->createConstantBitmap8kb();
    // If we need the big lookup table, create it.
    this->createGlobalBitmap512MB();

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