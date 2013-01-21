

// A network based plaintext queue.  Should be threadsafe.
// Listens on a given port & adds words to the queue if there is space.
// Pauses if there is not space - this should eventually lead to network throttling.

#include <stdint.h>
#include <queue>
#include <string>
#include <vector>


// Use boost mutexes to restrict access.
#include <boost/thread/mutex.hpp>
#include <boost/asio.hpp>
#include <boost/thread.hpp>

#include "CH_Common/CHHiresTimer.h"

// Max number of IO threads
#define MAX_IO_THREADS 5

// Forward define the various classes
class CHNetworkPlainQueue;
class CHNetworkPlainQueueSession;
class CHNetworkPlainQueueInstance;


class CHNetworkPlainQueue {
public:
    // Constructor.  Takes the network port as an argument and starts the server.
    CHNetworkPlainQueue (uint16_t networkPort);
    
    // Get the number of plains in the queue.
    uint64_t getNumberPlainsInQueue();
    
    // Gets the total number of plains passed out so far.
    uint64_t getNumberPlainsProcessed();
    
    // Adds the given vector of words to the array.
    void addPlainsToQueue(std::vector <std::vector<uint8_t> >);
    
    // Returns the requested number of plains, or the maximum present.
    std::vector <std::vector<uint8_t> > getNextNPlains(uint32_t);
    
    // Network init & teardown
    void startNetwork();
    void stopNetwork();
    
    std::deque <std::vector<uint8_t> > plainQueue;
    
private:
    
    uint16_t portNumber;
    
    // Data structure to contain the plains.
    
    // Counter for the number of plans we have passed out.
    uint64_t numberPlainsProcessed;
    
    // == Threading Section ==
    boost::mutex queueMutex;
    
    boost::asio::io_service io_service;
    CHNetworkPlainQueueInstance *NetworkServer;

    boost::thread *ioThreads[MAX_IO_THREADS];
    
};





using boost::asio::ip::tcp;

class CHNetworkPlainQueueSession {
public:

    CHNetworkPlainQueueSession(boost::asio::io_service& io_service,
            CHNetworkPlainQueue* CHNetworkPlainQueuePtr)
    : socket_(io_service) {
        this->networkPlainQueue = CHNetworkPlainQueuePtr;
    }

    tcp::socket& socket() {
        return socket_;
    }

    void start();

    void handle_read(const boost::system::error_code& error, size_t bytes_transferred);

    void handle_write(const boost::system::error_code& error);

private:
    
    // Reads characters out of the buffer and adds them to the main queue.
    void addWordsToQueue();
    
    tcp::socket socket_;
    enum {
        max_length = 1024*1024*10 // 10MB
    };
    uint8_t data_[max_length];

    char hostIpAddress[1024];

    CHNetworkPlainQueue* networkPlainQueue;
    
    std::vector <uint8_t> charBuffer;
};

class CHNetworkPlainQueueInstance {
public:

    CHNetworkPlainQueueInstance(boost::asio::io_service& io_service, short port, 
            CHNetworkPlainQueue* CHNetworkPlainQueuePtr)
    : io_service_(io_service),
    acceptor_(io_service, tcp::endpoint(tcp::v4(), port)) {
        printf("CHNetworkPlainQueueInstance::CHNetworkPlainQueueInstance()\n");
        this->networkPlainQueue = CHNetworkPlainQueuePtr;
        CHNetworkPlainQueueSession* new_session = new CHNetworkPlainQueueSession(io_service_, this->networkPlainQueue);
        acceptor_.async_accept(new_session->socket(),
                boost::bind(&CHNetworkPlainQueueInstance::handle_accept, this, new_session,
                boost::asio::placeholders::error));
    }

    void handle_accept(CHNetworkPlainQueueSession* new_session,
            const boost::system::error_code& error) {
        printf("In server::handle_accept()\n");
        if (!error) {
            new_session->start();
            new_session = new CHNetworkPlainQueueSession(io_service_, this->networkPlainQueue);
            acceptor_.async_accept(new_session->socket(),
                    boost::bind(&CHNetworkPlainQueueInstance::handle_accept, this, new_session,
                    boost::asio::placeholders::error));
        } else {
            delete new_session;
        }
    }

private:
    boost::asio::io_service& io_service_;
    tcp::acceptor acceptor_;
    CHNetworkPlainQueue* networkPlainQueue;
};