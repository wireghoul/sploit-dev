// Implementation of the CHNetworkPlainQueue class for word handling

#include "NetworkWords_Common/CHNetworkPlainQueue.h"

#define UNIT_TEST 1


void RunIoService(boost::asio::io_service* io_service_param) {
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


CHNetworkPlainQueue::CHNetworkPlainQueue (uint16_t networkPort) {
    
    printf("CHNetworkPlainQueue::CHNetworkPlainQueue(%d)\n", networkPort);
    this->portNumber = networkPort;
    
    // Init various variables to zero/null
    this->numberPlainsProcessed = 0;
    this->NetworkServer = NULL;
}

uint64_t CHNetworkPlainQueue::getNumberPlainsInQueue() {
    return this->plainQueue.size();
}

uint64_t CHNetworkPlainQueue::getNumberPlainsProcessed() {
    return this->numberPlainsProcessed;
}


void CHNetworkPlainQueue::addPlainsToQueue(std::vector<std::vector<uint8_t> > plainsToAdd) {
    uint32_t i;
        
    this->queueMutex.lock();
    
    for (i = 0; i < plainsToAdd.size(); i++) {
        this->plainQueue.push_back(plainsToAdd[i]);
    }
    this->queueMutex.unlock();
}

std::vector <std::vector<uint8_t> > CHNetworkPlainQueue::getNextNPlains(uint32_t plainsToGet) {
    std::vector<std::vector<uint8_t> > dataToReturn;
    dataToReturn.reserve(plainsToGet);

    this->queueMutex.lock();
    for (uint32_t i = 0; i < plainsToGet; i++) {
        dataToReturn.push_back(this->plainQueue.front());
        this->plainQueue.pop_front();
    }
    this->queueMutex.unlock();

    return dataToReturn;
}


void CHNetworkPlainQueue::startNetwork() {
    printf("CHNetworkPlainQueue::startNetwork()\n");
    
    int i;

    // Only create the network server if it's not already present.
    if (!this->NetworkServer) {
        this->NetworkServer = new CHNetworkPlainQueueInstance(this->io_service, this->portNumber, this);
    }
    
    // Launch all the network IO threads
    for (i = 0; i < MAX_IO_THREADS; i++) {
        this->ioThreads[i] = new boost::thread(RunIoService, &this->io_service);
    }
}


// Stop the io_service instances, and wait for all the threads to return.
void CHNetworkPlainQueue::stopNetwork() {

    int i;

    this->io_service.stop();
    for (i = 0; i < MAX_IO_THREADS; i++) {
        this->ioThreads[i]->join();
    }
}




// CHNetworkServerSession functions
void CHNetworkPlainQueueSession::start() {
    printf("In CHNetworkPlainQueueSession::start()\n");

    sprintf(this->hostIpAddress, "%s", socket_.remote_endpoint().address().to_string().c_str());

    socket_.async_read_some(boost::asio::buffer(data_, max_length),
            boost::bind(&CHNetworkPlainQueueSession::handle_read, this,
            boost::asio::placeholders::error,
            boost::asio::placeholders::bytes_transferred));
}



void CHNetworkPlainQueueSession::handle_read(const boost::system::error_code& error,
            size_t bytes_transferred) {
    //printf("In session::handle_read()\n");
    //printf("Buffer (%d): %c\n", bytes_transferred, data_[0]);

    if (!error) {
        if (bytes_transferred > 0) {
            std::copy((uint8_t*)&data_[0], ((uint8_t*) &data_[0]) + bytes_transferred, std::back_inserter(this->charBuffer));
        }
        if (this->charBuffer.size() > 100000) {
            this->addWordsToQueue();
        }

        socket_.async_read_some(boost::asio::buffer(data_, max_length),
            boost::bind(&CHNetworkPlainQueueSession::handle_read, this,
            boost::asio::placeholders::error,
            boost::asio::placeholders::bytes_transferred));

    } else {
        // Report the disconnect
        printf("\n\nDSC: %s", this->hostIpAddress);
        this->addWordsToQueue();
        delete this;
    }
}


// This is the slow function.  Optimize here!
void CHNetworkPlainQueueSession::addWordsToQueue() {
    std::vector<uint8_t> wordToAdd;
    
    std::vector<std::vector<uint8_t> > vectorOfWords;
    
    std::vector<uint8_t>::iterator wordStart, wordStop;

    vectorOfWords.reserve(200000);

    // Loop through the plains, finding newlines & creating strings.
    wordStart = this->charBuffer.begin();
    // Cpr E!  Cpr E!  while(1)!  while(1)!
    wordToAdd.clear();
    // Start at the starting point and look for a newline.
    wordStop = wordStart;
    for (wordStop = wordStart; wordStop != this->charBuffer.end(); wordStop++) {
        if (*wordStop == '\n' || *wordStop == '\r' || (wordStop == this->charBuffer.end())) {
            //printf("Found newline!\n");
            wordToAdd.assign(wordStart, wordStop);
            //printf("Got word to add %s\n", wordToAdd.c_str());
            vectorOfWords.push_back(wordToAdd);
            wordToAdd.clear();
            if ((vectorOfWords.size() % 100000) == 0) {
                vectorOfWords.reserve(vectorOfWords.size() + 100000);
                printf("Resizing vector to %d\n", vectorOfWords.size() + 100000);
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
    
    this->networkPlainQueue->addPlainsToQueue(vectorOfWords);
    this->charBuffer.erase(this->charBuffer.begin(), wordStart);
}



#if UNIT_TEST

#include <unistd.h>

int main() {
    
    CHNetworkPlainQueue NetworkQueue(4444);
    
    NetworkQueue.startNetwork();

    for (int i = 0; i < 20; i++) {
        sleep(1);
        printf("Size of buffer: %d\n", NetworkQueue.getNumberPlainsInQueue());
    }

    while (NetworkQueue.getNumberPlainsInQueue()) {
        std::vector<std::vector<uint8_t> > data = NetworkQueue.getNextNPlains(1);
        data[0].push_back(0);
        printf("Plain: %s\n", &data[0][0]);
    }
    
}

#endif