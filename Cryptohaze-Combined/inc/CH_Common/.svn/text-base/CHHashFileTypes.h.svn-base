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

// CHHashFileTypes implements the prototype for the hash file functions.
// This includes reading in files, storing/providing hash lists, and reporting found passwords.

#ifndef _CHHASHFILETYPES_H
#define _CHHASHFILETYPES_H

#include "Multiforcer_Common/CHCommon.h"

// We need to include the mutex defines.
#if USE_BOOST_THREADS
        #include <boost/thread.hpp>
        #include <boost/thread/mutex.hpp>
#else
        #include <pthread.h>
#endif


#if USE_NETWORK
class CHNetworkClient;
#endif

class CHHashFileTypes {
#if USE_NETWORK
protected:
    CHNetworkClient *NetworkClient;
#endif
    char AddHexOutput;

    // Mutexes for access to the various hashfile functions.
    // As stl containers are NOT threadsafe, we cannot have multiple threads
    // in the workunits at any given point in time.
#if USE_BOOST_THREADS
    boost::mutex hashfileMutexBoost;
#else
    pthread_mutex_t hashfileMutexPThreads;
    pthread_mutexattr_t hashfileMutexPThreadsAttributes;
#endif

    // Mutex functions for the class
    void LockMutex() {
#if USE_BOOST_THREADS
        this->hashfileMutexBoost.lock();
#else
        pthread_mutex_lock(&this->hashfileMutexPThreads);
#endif
    }

    void UnlockMutex() {
#if USE_BOOST_THREADS
        this->hashfileMutexBoost.unlock();
#else
        pthread_mutex_unlock(&this->hashfileMutexPThreads);
#endif
    }

public:

    CHHashFileTypes() {
        // Initialize pthread mutexes if needed.
#if !USE_BOOST_THREADS
        pthread_mutexattr_init(&this->hashfileMutexPThreadsAttributes);
        pthread_mutex_init(&this->hashfileMutexPThreads, &this->hashfileMutexPThreadsAttributes);
#endif
    }

    // Opens a hash file of the appropriate format.
    // Returns 1 on successful load, 0 on failure.
    virtual int OpenHashFile(char *filename) = 0;

    // Exports a list of uncracked hashes in sorted order.
    // Each element is a hash string (just the hash).
    virtual unsigned char *ExportUncrackedHashList() = 0;

    // Used to report a found password.
    // Returns 1 for successfully added, 0 for not added properly.
    virtual int ReportFoundPassword(unsigned char *Hash, unsigned char *Password) = 0;

    //  Prints the entire list of found hashes.
    virtual void PrintAllFoundHashes() = 0;

    // Prints only the newly found hashes that haven't been reported.
    virtual void PrintNewFoundHashes() = 0;

    // Outputs all found hashes to the specified file.
    // Returns 1 on success, 0 on failure.
    virtual int OutputFoundHashesToFile() = 0;

    virtual void SetFoundHashesOutputFilename(char *filename) = 0;

    // Outputs all unfound hashes to the specified file.
    // Returns 1 on success, 0 on failure.
    virtual int OutputUnfoundHashesToFile(char *filename) = 0;

    // Returns the total number of hashes loaded.
    virtual unsigned long GetTotalHashCount() = 0;

    // Do what they imply.
    virtual unsigned long GetCrackedHashCount() = 0;
    virtual unsigned long GetUncrackedHashCount() = 0;

    virtual int GetHashLength() {
        return 0;
    }
    virtual int GetSaltLength() {
        return 0;
    }

    virtual void SetAddHexOutput(char newAddHexOutput) {
        this->AddHexOutput = newAddHexOutput;
    }


    // Import a number of hashes from a remote system.
    virtual void importHashListFromRemoteSystem(unsigned char *hashData, uint32_t numberHashes) {
        return;
    }

#if USE_NETWORK
    virtual void submitFoundHashToNetwork(unsigned char *Hash, unsigned char *Password) {
        return;
    }
    virtual void setNetworkClient(CHNetworkClient *newNetworkClient) {
        this->NetworkClient = newNetworkClient;
    }

#endif

};


#endif
