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
#ifndef __MFNWORKUNITWORDLIST_H
#define __MFNWORKUNITWORDLIST_H

/**
 * This workunit class is designed to handle the demands of wordlist based
 * cracking.  It will communicate with the wordlist class and receive additional
 * data, packing it into workunits.  Perhaps this could be improved by combining
 * the wordlist class and the workunit class...
 *
 * In any case, wordlist entries get sucked into workunits, which are handed
 * out with start/end values correpsonding to the number of words in the data
 * blob.
 *
 * This is heavily based on the robust workunit class, as workunits for wordlist
 * processing should be retried as needed.
 */

#include <deque>
#include <vector>
#include <list>
#include <string>
#include <boost/date_time/posix_time/posix_time.hpp>


#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/asio.hpp>

#include "MFN_Common/MFNWorkunitRobust.h"
#include "CH_Common/CHHiresTimer.h"
#include "MFN_Common/MFNWorkunit.pb.h"
#include "MFN_Common/MFNDebugging.h"


/**
 * First pass at a structure to contain password lists sorted by length.  This
 * keeps corresponding lengths with each vector.  The number of passwords stored
 * in each vector is also kept.
 * 
 * Passwords are optionally padded in storage with 0x80 - this is factored into
 * their storage vector, but is not reflected in length.
 * 
 * Each length stores passwords in just as many blocks as needed.  So, for
 * length 4, password 0 is stored in block 0.  For password length 8, password
 * 0 is stored in block 0,1.  Etc.
 * 
 * They're stored in "native order" - they will be swizzled to the GPU format
 * for the hash at load.
 */

// Max block length - words are block length * 4.
#define MAX_WORDLIST_BLOCKS 32
#define MAX_WORD_LENGTH 128

// Max pending words before the streams hang.
#define MAX_PENDING_WORD_COUNT 200000000

// How many words per workunit
#define WORDS_PER_WORKUNIT 10000000

// How big each workunit can be.
#define MAX_BYTES_PER_WORDLIST (120*1024*1024)

// Max age of an set of words in seconds before it's run
#define MAX_WORD_AGE 60

// Max pregenerated workunit count
#define MAX_PENDING_WORKUNITS 100

typedef struct {
    // True if passwords are 0x80 padded.
    uint8_t passwordsArePadded;
    // How many passwords are in each vector.
    uint64_t passwordsCurrentlyLoaded[MAX_WORDLIST_BLOCKS];
    // Password lengths for each position.
    std::vector<uint8_t> passwordLengthBytes[MAX_WORDLIST_BLOCKS];
    // Password data by block.
    std::vector<uint32_t> passwordData[MAX_WORDLIST_BLOCKS];
    // Last exported time, in seconds.
    uint64_t lastExportedTime[MAX_WORDLIST_BLOCKS];
} wordlistByBlockSize;

using boost::asio::ip::tcp;

// Network queue class.
#define MFN_WORKUNIT_WORDLIST_MAX_IO_THREADS 10

// Default network port
#define MFN_WORKUNIT_DEFAULT_NETWORK_PORT 4444

class MFNWorkunitWordlistSession;
class MFNWorkunitWordlistInstance;


class MFNWorkunitWordlist : public MFNWorkunitRobust {
protected:
    /**
     * Create more workunits.  This appends more pending workunits to the 
     * queue.  It will not create units "past the end," but will create
     * up to the number requested if needed.
     * 
     * @parm numberToAdd How many more to create.
     */
    void CreateMorePendingWorkunits(uint32_t numberToAdd);
    
    void updateWordCount();

    uint16_t networkPort;

    // == Threading Section ==
    boost::mutex queueMutex;

    boost::asio::io_service io_service;
    MFNWorkunitWordlistInstance *NetworkServer;

    boost::thread *ioThreads[MFN_WORKUNIT_WORDLIST_MAX_IO_THREADS];

    // The wordlist holder.
    wordlistByBlockSize wordlistBlocks;
    boost::mutex wordlistBlockMutex;
    
    // Total words loaded.
    uint64_t totalQueuedWords;
    
    // Last workunit ID created
    uint64_t lastWorkunitIDAssigned;
    
public:

    MFNWorkunitWordlist();
    ~MFNWorkunitWordlist();

    virtual int CreateWorkunits(uint64_t NumberOfPasswords, uint8_t BitsPerUnit,
        uint8_t PasswordLength);

    void PrintInternalState();

    /**
     * Sets the network port to use for incoming wordlists.
     */
    void SetWordlistNetworkPort(uint16_t newNetworkPort) {
        this->networkPort = newNetworkPort;
    }
    
    // Start & stop network wordlist use.
    void StartNetwork();
    void StopNetwork();
    
    uint64_t GetTotalWordsQueued() {
        return this->totalQueuedWords;
    }
    
    // Adds a bunch of newly generated words to the queue.
    void addWordsToQueue(wordlistByBlockSize &newWords);
    
    uint64_t GetNumberOfWorkunits() {
        return this->NumberOfWorkunitsTotal + 1;
    }
    
    uint64_t GetNumberOfCompletedWorkunits() {
        return this->NumberOfWorkunitsCompleted;
    }

    

    // Modified workunit return - always delay.
    struct MFNWorkunitRobustElement GetNextWorkunit(uint32_t NetworkClientId);
};


// Session class for the network - handles the bulk of the work.
class MFNWorkunitWordlistSession {
public:

    MFNWorkunitWordlistSession(boost::asio::io_service& io_service,
            MFNWorkunitWordlist* MFNWorkunitWordlistPtr)
    : socket_(io_service) {
       trace_printf("MFNWorkunitWordlistSession::MFNWorkunitWordlistSession()\n");
       this->networkPlainQueue = MFNWorkunitWordlistPtr;
    }

    tcp::socket& socket() {
        return socket_;
    }

    void start();

    void handle_read(const boost::system::error_code& error, size_t bytes_transferred);

private:

    // Reads characters out of the buffer and adds them to the main queue.
    void addWordsToQueue();

    tcp::socket socket_;
    enum {
        max_length = 1024*1024*10 // 10MB
    };
    uint8_t data_[max_length];

    char hostIpAddress[1024];

    MFNWorkunitWordlist* networkPlainQueue;

    std::vector <uint8_t> charBuffer;
};

// Entirely a header based class - no need for implementation in the source.
// Pogos a network accept into a session.
class MFNWorkunitWordlistInstance {
public:

    MFNWorkunitWordlistInstance(boost::asio::io_service& io_service, short port,
            MFNWorkunitWordlist* MFNWorkunitWordlistPtr)
    : io_service_(io_service),
    acceptor_(io_service, tcp::endpoint(tcp::v4(), port)) {
        trace_printf("MFNWorkunitWordlistInstance::MFNWorkunitWordlistInstance()\n");
        this->networkPlainQueue = MFNWorkunitWordlistPtr;
        MFNWorkunitWordlistSession* new_session = new MFNWorkunitWordlistSession(io_service_, this->networkPlainQueue);
        acceptor_.async_accept(new_session->socket(),
                boost::bind(&MFNWorkunitWordlistInstance::handle_accept, this, new_session,
                boost::asio::placeholders::error));
    }

    void handle_accept(MFNWorkunitWordlistSession* new_session,
            const boost::system::error_code& error) {
        trace_printf("In instance::handle_accept()\n");
        if (!error) {
            new_session->start();
            new_session = new MFNWorkunitWordlistSession(io_service_, this->networkPlainQueue);
            acceptor_.async_accept(new_session->socket(),
                    boost::bind(&MFNWorkunitWordlistInstance::handle_accept, this, new_session,
                    boost::asio::placeholders::error));
        } else {
            delete new_session;
        }
    }

private:
    boost::asio::io_service& io_service_;
    tcp::acceptor acceptor_;
    MFNWorkunitWordlist* networkPlainQueue;
};


#endif

