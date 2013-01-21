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


// Implementation for the wordlist workunit type...

#include <deque>
#include <list>
#include <vector>
#include <string>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "MFN_Common/MFNWorkunitWordlist.h"
//#define TRACE_PRINTF 1
#include "MFN_Common/MFNDebugging.h"
#include "MFN_Common/MFNDefines.h"

extern void PrintRobustWorkunit(struct MFNWorkunitRobustElement ElementToPrint);

static void dumpWordlistStruct(wordlistByBlockSize *words) {
    
    printf("======== Dumping Wordlist ========\n");

    for (int i = 0; i < MAX_WORDLIST_BLOCKS; i++) {
        printf("Block %d\n", i);
        printf("Number passwords: %d\n", words->passwordsCurrentlyLoaded[i]);
        for (int j = 0; j < words->passwordsCurrentlyLoaded[i]; j++) {
            printf("%d: ", words->passwordLengthBytes[i][j]);
            for (int k = 0; k < words->passwordLengthBytes[i][j]; k++) {
                printf("%c", (words->passwordData[i][(j * (i + 1)) + (k / 4)] >> ((k % 4) * 8)) & 0xff);
            }
            printf("\n");
        }
    }
    
}


MFNWorkunitWordlist::MFNWorkunitWordlist() {
    trace_printf("MFNWorkunitWordlist::MFNWorkunitWordlist()\n");
    this->networkPort = MFN_WORKUNIT_DEFAULT_NETWORK_PORT;
    this->NetworkServer = NULL;
    this->wordlistBlocks.passwordsArePadded = 0;
    memset(this->wordlistBlocks.passwordsCurrentlyLoaded, 0, sizeof(this->wordlistBlocks.passwordsCurrentlyLoaded));
    memset(this->wordlistBlocks.lastExportedTime, 0, sizeof(this->wordlistBlocks.lastExportedTime));
    this->totalQueuedWords = 0;
    this->lastWorkunitIDAssigned = 0;
    this->StartNetwork();
}

MFNWorkunitWordlist::~MFNWorkunitWordlist() {
    trace_printf("MFNWorkunitWordlist::~MFNWorkunitWordlist()\n");
}

void MFNWorkunitWordlist::CreateMorePendingWorkunits(uint32_t numberToAdd) {
    trace_printf("MFNWorkunitRobust::CreateMorePendingWorkunits(%d)\n", numberToAdd);

    /**
     * Create workunits with the wordlist contents in the following manner:
     * - For each array of word blocks, pull the top N off and push them
     * into a workunit of the given block length.
     * - If the block is beyond the given age, add it anyway - even if it's only
     * a few words.
     * 
     */
    
    MFNWorkunitRobustElement NewWorkunit;
    
    this->ClearWorkunit(NewWorkunit);
    
    NewWorkunit.IsValid = 1;
    
    
    // Lock the mutex - in the wordlist stuff.
    this->wordlistBlockMutex.lock();
    
    // Take the chunks of words & throw them into the workunits.
    // TODO: Obey the above.  For now, just cram everything present in.
    for (int i = 0; i < MAX_WORDLIST_BLOCKS; i++) {
        // If there are any words loaded, add them to a WU.
        if (this->wordlistBlocks.passwordsCurrentlyLoaded[i]) {
            //printf("%d passwords in block %d\n", this->wordlistBlocks.passwordsCurrentlyLoaded[i], i);
            // Words to put in this workunit.  Size divided by (blocks * 4)
            uint32_t wordsForWorkunit = MAX_BYTES_PER_WORDLIST / ((i + 1) * 4);
            if (this->wordlistBlocks.passwordsCurrentlyLoaded[i] < wordsForWorkunit) {
                wordsForWorkunit = this->wordlistBlocks.passwordsCurrentlyLoaded[i];
            }
            //printf("Block %d: Copying %d words.\n", i, this->wordlistBlocks.passwordsCurrentlyLoaded[i]);
            // Set the block length
            NewWorkunit.WordBlockLength = i + 1;
            NewWorkunit.NumberWordsLoaded = wordsForWorkunit;
            NewWorkunit.WordLengths.clear();
            NewWorkunit.WordlistData.clear();
            
            // Reduce count by the number we're copying.
            this->wordlistBlocks.passwordsCurrentlyLoaded[i] -= wordsForWorkunit;

            // Copy in the lengths from the main array.
            NewWorkunit.WordLengths.assign(
                    this->wordlistBlocks.passwordLengthBytes[i].begin(),
                    this->wordlistBlocks.passwordLengthBytes[i].begin() + wordsForWorkunit);
            // Remove the entries from the front of the main array.
            this->wordlistBlocks.passwordLengthBytes[i].erase(
                    this->wordlistBlocks.passwordLengthBytes[i].begin(),
                    this->wordlistBlocks.passwordLengthBytes[i].begin() + wordsForWorkunit);
            //printf("Post-copy passwordLengthBytes size: %d\n", this->wordlistBlocks.passwordLengthBytes[i].size());

            
            // Copy in the data from the main array.
            NewWorkunit.WordlistData.assign(
                    this->wordlistBlocks.passwordData[i].begin(),
                    this->wordlistBlocks.passwordData[i].begin() +
                    (wordsForWorkunit * NewWorkunit.WordBlockLength));
            // Remove the entries from the front of the main array.
            this->wordlistBlocks.passwordData[i].erase(
                    this->wordlistBlocks.passwordData[i].begin(),
                    this->wordlistBlocks.passwordData[i].begin() +
                    (wordsForWorkunit * NewWorkunit.WordBlockLength));
            //printf("Post-copy passwordData size: %d\n", this->wordlistBlocks.passwordData[i].size());
            
            // Set the WU ID & increment.
            NewWorkunit.WorkUnitID = lastWorkunitIDAssigned++;
        
            this->pendingWorkunits.push_back(NewWorkunit);
            this->NumberOfWorkunitsTotal++;
        }
    }
    
    // Update the total word count after popping data off the queue.
    // Hopefully, this will resolve blocked threads... threads...
    this->updateWordCount();

    this->wordlistBlockMutex.unlock();
}

int MFNWorkunitWordlist::CreateWorkunits(uint64_t NumberOfPasswords, uint8_t BitsPerUnit, uint8_t PasswordLength) {
    trace_printf("MFNWorkunitRobust::CreateWorkunits(%lu, %u, %u)\n", NumberOfPasswords, BitsPerUnit, PasswordLength);

    this->ClearAllInternalState();

    this->NumberOfWorkunitsTotal = 0; //0xffffffffffffffff;

    this->WorkunitClassInitialized = 1;
    this->CurrentPasswordLength = PasswordLength;

    // Start the execution timer
    this->WorkunitTimer.start();
    this->LastStateSaveTime = 0;

    return 1;
}



void MFNWorkunitWordlist::PrintInternalState() {
    trace_printf("MFNWorkunitWordlist::PrintInternalState()\n");
    int i;

    printf("Number WU total: %lu\n", this->NumberOfWorkunitsTotal);
    printf("Number WU completed: %lu\n", this->NumberOfWorkunitsCompleted);
    printf("Number WU left: %lu\n", this->pendingWorkunits.size());
    printf("Number WU inflight: %lu\n", this->assignedWorkunits.size());
    printf("Active client IDs: ");
    for (i = 0 ; i < this->inUseClientIds.size(); i++) {
        printf("%d, ", this->inUseClientIds[i]);
    }
    printf("\n");
}

void RunMFNNetworkWordlistIoService(boost::asio::io_service* io_service_param) {
    for (;;) {
    try
    {
      io_service_param->run();
      break; // run() exited normally
    }
    catch (boost::system::system_error& e)
    {
        printf("\n\nGOT EXCEPTION IN RunIoService!!!\n");
        printf("Exception data: %s\n", e.what());
        io_service_param->reset();

      // Deal with exception as appropriate.
    }
  }
}


void MFNWorkunitWordlist::StartNetwork() {
    // Only create the network server if it's not already present.
    if (!this->NetworkServer) {
        this->NetworkServer = new MFNWorkunitWordlistInstance(this->io_service, this->networkPort, this);
    }
    
    // Launch all the network IO threads
    for (int i = 0; i < MFN_WORKUNIT_WORDLIST_MAX_IO_THREADS; i++) {
        this->ioThreads[i] = new boost::thread(RunMFNNetworkWordlistIoService, &this->io_service);
    }
}

void MFNWorkunitWordlist::StopNetwork() {
    this->io_service.stop();
    for (int i = 0; i < MFN_WORKUNIT_WORDLIST_MAX_IO_THREADS; i++) {
        this->ioThreads[i]->join();
    }
}


// CHNetworkServerSession functions
void MFNWorkunitWordlistSession::start() {
    sprintf(this->hostIpAddress, "%s", socket_.remote_endpoint().address().to_string().c_str());

    socket_.async_read_some(boost::asio::buffer(data_, max_length),
            boost::bind(&MFNWorkunitWordlistSession::handle_read, this,
            boost::asio::placeholders::error,
            boost::asio::placeholders::bytes_transferred));
}



void MFNWorkunitWordlistSession::handle_read(const boost::system::error_code& error,
            size_t bytes_transferred) {
    //printf("In session::handle_read()\n");
    //printf("Buffer (%d): %c\n", bytes_transferred, data_[0]);

    if (!error) {
        
        // Sleep if the list is too big already.
        while (this->networkPlainQueue->GetTotalWordsQueued() > MAX_PENDING_WORD_COUNT) {
            //printf("Sleeping for wordcount...\n");
            CHSleep(1);
        }
        
        if (bytes_transferred > 0) {
            std::copy((uint8_t*)&data_[0], ((uint8_t*) &data_[0]) + bytes_transferred, std::back_inserter(this->charBuffer));
        }
        if (this->charBuffer.size() > 10000000) {
            this->addWordsToQueue();
        }

        socket_.async_read_some(boost::asio::buffer(data_, max_length),
            boost::bind(&MFNWorkunitWordlistSession::handle_read, this,
            boost::asio::placeholders::error,
            boost::asio::placeholders::bytes_transferred));

    } else {
        // Report the disconnect
        //printf("\n\nDSC: %s\n", this->hostIpAddress);
        fflush(stdout);
        this->addWordsToQueue();
        delete this;
    }
}



// This is the slow function.  Optimize here!
void MFNWorkunitWordlistSession::addWordsToQueue() {
    std::vector<uint8_t> wordToAdd;
    
    const char addPadding = 1;
    
    wordlistByBlockSize localWords;
    
    std::vector<uint8_t>::iterator wordStart, wordStop;
    
    localWords.passwordsArePadded = 0;
    memset(localWords.passwordsCurrentlyLoaded, 0, sizeof(localWords.passwordsCurrentlyLoaded));
    memset(localWords.lastExportedTime, 0, sizeof(localWords.lastExportedTime));

    // Loop through the plains, finding newlines & creating strings.
    wordStart = this->charBuffer.begin();

    wordToAdd.clear();
    wordToAdd.reserve(1024);
    // Start at the starting point and look for a newline.
    wordStop = wordStart;
    for (wordStop = wordStart; wordStop != this->charBuffer.end(); wordStop++) {
        // Word length in 32-bit blocks
        uint8_t wordLengthBlocks;
        // Actual word length
        uint8_t wordLength;
        uint8_t wordLengthPadded;
        
        if (*wordStop == '\n' || *wordStop == '\r' || (wordStop == this->charBuffer.end())) {
            //printf("Found newline!\n");
            // Clear out the reserved space.
            memset(&wordToAdd[0], 0, wordToAdd.capacity());
            
            // Assign the word to the vector.
            wordToAdd.assign(wordStart, wordStop);
            
            // Only add words with non-zero length.
            if (/*wordToAdd.size() && */(wordToAdd.size() < MAX_WORD_LENGTH)) {
                wordLength = wordToAdd.size();
                wordLengthPadded = wordLength;
                
                if (addPadding) {
                    wordToAdd.push_back(0x80);
                    wordLengthPadded++;
                }
                
                wordLengthBlocks = wordLengthPadded / 4;
                // If it's an even multiple of 4, use the previous block size.
                if ((wordLengthPadded % 4) == 0) {
                    wordLengthBlocks--;
                }
                
                //printf("Word length: %d\n", wordToAdd.size());
                //printf("Bucket: %d\n", wordLengthBlocks);
                uint32_t *wordPointer = (uint32_t *)&wordToAdd[0];

                for (int i = 0; i < (wordLengthBlocks + 1); i++) {
                    localWords.passwordData[wordLengthBlocks].push_back(wordPointer[i]);
                }
                localWords.passwordLengthBytes[wordLengthBlocks].push_back(wordLength);
                localWords.passwordsCurrentlyLoaded[wordLengthBlocks]++;

                //wordToAdd.push_back(0);
                //printf("Got word to add %s\n", (char *)&wordToAdd[0]);
                wordToAdd.clear();
            }
            while (wordStop != this->charBuffer.end() && (*wordStop == '\n' || *wordStop == '\r')) {
                wordStop++;
            }
            wordStart = wordStop;
            if (wordStop == this->charBuffer.end()) {
                break;
            }
        }
    }
    
    this->charBuffer.erase(this->charBuffer.begin(), wordStart);
    
    //dumpWordlistStruct(&localWords);
    this->networkPlainQueue->addWordsToQueue(localWords);
}


// Mutex should be locked before this is called.
void MFNWorkunitWordlist::updateWordCount() {
    uint64_t totalWords = 0;
    for (int i = 0; i < MAX_WORDLIST_BLOCKS; i++) {
        totalWords += this->wordlistBlocks.passwordsCurrentlyLoaded[i];
    }
    this->totalQueuedWords = totalWords;
}

void MFNWorkunitWordlist::addWordsToQueue(wordlistByBlockSize &newWords) {
    this->wordlistBlockMutex.lock();
    wordlistByBlockSize *globalWords = 
            &this->wordlistBlocks;
    for (int i = 0; i < MAX_WORDLIST_BLOCKS; i++) {
        if (newWords.passwordsCurrentlyLoaded[i]) {
            //printf("Copying %d passwords to global\n", localWords.passwordsCurrentlyLoaded[i]);
            // Add number of passwords.
            globalWords->passwordsCurrentlyLoaded[i] +=
                    newWords.passwordsCurrentlyLoaded[i];
            // Add length data
            globalWords->passwordLengthBytes[i].reserve(
                    globalWords->passwordLengthBytes[i].size() + 
                    newWords.passwordLengthBytes[i].size());
            globalWords->passwordLengthBytes[i].insert( 
                    globalWords->passwordLengthBytes[i].end(), 
                    newWords.passwordLengthBytes[i].begin(), 
                    newWords.passwordLengthBytes[i].end() );
            // Add the actual words
            globalWords->passwordData[i].reserve(
                    globalWords->passwordData[i].size() + 
                    newWords.passwordData[i].size());
            globalWords->passwordData[i].insert( 
                    globalWords->passwordData[i].end(), 
                    newWords.passwordData[i].begin(), 
                    newWords.passwordData[i].end() );
        }
    }
    
    // Update the total word count
    this->updateWordCount();
    //dumpWordlistStruct(&this->wordlistBlocks);
    this->wordlistBlockMutex.unlock();
}

struct MFNWorkunitRobustElement MFNWorkunitWordlist::GetNextWorkunit(uint32_t ClientId) {
    trace_printf("MFNWorkunitWordlist::GetNextWorkunit(%u)\n", ClientId);

    struct MFNWorkunitRobustElement Workunit;

    this->workunitMutexBoost.lock();
    
    // Check to see if we need to make more workunits.
    if (this->pendingWorkunits.size() < MFN_WORKUNIT_MIN_PENDING_WUS) {
        this->CreateMorePendingWorkunits(MFN_WORKUNIT_WU_REFILL_SIZE);
    }

    // Check to see if there are valid workunits left.
    if (this->pendingWorkunits.size() == 0) {
        // If not, return a unit with isValid = 0.
        if (this->DebugOutput) {
            printf("pendingWorkunits.size() == 0; returning.\n");
        }
        memset(&Workunit, 0, sizeof(MFNWorkunitRobustElement));
        Workunit.Flags = WORKUNIT_DELAY;
        this->workunitMutexBoost.unlock();
        if (this->DebugOutput) {
            PrintRobustWorkunit(Workunit);
        }
        return Workunit;
    }

    // We still have workunits left.

    // Get the next waiting workunit from the main queue.
    Workunit = this->pendingWorkunits.front();
    this->pendingWorkunits.pop_front();

    if (this->DebugOutput) {
        printf("Popped WU ID %lu\n", Workunit.WorkUnitID);
    }

    // Set some variables we can make use of.
    Workunit.IsAssigned = 1;
    Workunit.WorkunitRequestedTimestamp = this->WorkunitTimer.getElapsedTime();
    Workunit.ClientId = ClientId;

    // Add the workunit to the in-flight queue.
    this->assignedWorkunits.push_back(Workunit);
    if (this->DebugOutput) {
        printf("In flight WUs: %lu\n", this->assignedWorkunits.size());
    }  

    this->workunitMutexBoost.unlock();
    this->WriteSaveState(0);
    if (this->DebugOutput) {
        PrintRobustWorkunit(Workunit);
    }
    return Workunit;
}

//#define WORDLIST_WU_UNIT_TEST 1

#if WORDLIST_WU_UNIT_TEST

int main() {
    printf("MFNWorkunitWordlist Unit Test!\n");
    int i;

    MFNWorkunitWordlist *WorkunitClass;
    MFNWorkunitRobustElement WU;
    
    uint32_t clientId;

    WorkunitClass = new MFNWorkunitWordlist();
    
    WorkunitClass->CreateWorkunits(0, 32, 16);

    WorkunitClass->EnableDebugOutput();
    
    WorkunitClass->StartNetwork();
    
    clientId = WorkunitClass->GetClientId();
    
    while (1) {
        sleep(1);
        printf("Words: %lu\n", WorkunitClass->GetTotalWordsQueued());
        WU = WorkunitClass->GetNextWorkunit(clientId);
        printf("Got workunit: ID %d, block length %d\n", WU.WorkUnitID, WU.WordBlockLength);
        WorkunitClass->SubmitWorkunit(WU);
    }
    
    
    //Workunit->LoadStateFromFile("resumeFile.chr");
    //Workunit->PrintInternalState();


    //Workunit->SetResumeFile("resumeFile.chr");

}



#endif
