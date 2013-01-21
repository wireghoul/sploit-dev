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

// This is a class that implements the network server functionality for
// the network features of the multiforcer.

#ifndef __CHNETWORKSERVER_H__
#define __CHNETWORKSERVER_H__

#include "Multiforcer_Common/CHCommon.h"

// Include classes we need.
#include "Multiforcer_CUDA_host/CHCommandLineData.h"
#include "CH_Common/CHCharset.h"
#include "CH_Common/CHWorkunit.h"
#include "CH_Common/CHWorkunitRobust.h"
#include "CH_Common/CHWorkunitNetwork.h"
#include "CH_Common/CHHashFileTypes.h"

#include "Multiforcer_Common/CHNetworkCommon.h"





class CHNetworkServer {
public:
    CHNetworkServer(int port);


    // Starts up the network functionality.  This brings things online.
    virtual int startNetwork();
    // Tear things down.
    virtual int stopNetwork();

    // Set the various classes this will need to communicate with
    virtual void setCommandLineData(CHCommandLineData *NewCommandLineData);
    virtual void setCharset(CHCharset *NewCharset);
    virtual void setWorkunit(CHWorkunitBase *NewWorkunit);
    virtual void setHashFile(CHHashFileTypes *NewHashFile);
    virtual void setDisplay(CHDisplay *NewDisplay);
    virtual void setPasswordLength(int);
    virtual int  getPasswordLength();
    virtual void setHashTypeId(int);
    virtual int  getHashTypeId();


protected:
    // TCP port number that will be used for this connection.
    int portNumber;
    
    // Current password length.
    int passwordLength;

    int hashTypeId;

    // This structure contains the various other class pointers of things
    // that have useful information we may want.
    struct classContainerStruct classContainer;

    boost::asio::io_service io_service;
    CHNetworkServerInstance *NetworkServer;

    boost::thread *ioThreads[MAX_IO_THREADS];


};



using boost::asio::ip::tcp;

class CHNetworkServerSession {
public:

    CHNetworkServerSession(boost::asio::io_service& io_service,
            struct classContainerStruct newClassContainer)
    : socket_(io_service) {
        this->classContainer = newClassContainer;
        this->currentAction = 0;
        this->displayThreadId = 0;
    }

    tcp::socket& socket() {
        return socket_;
    }

    void start();

    void handle_read(const boost::system::error_code& error, size_t bytes_transferred);

    void handle_write(const boost::system::error_code& error);

private:
    tcp::socket socket_;
    struct classContainerStruct classContainer;
    enum {
        max_length = 1024
    };
    char data_[max_length];
    unsigned char *hashList;
    char *charset;
    int currentAction; // What we are doing currently

    int displayThreadId;
    char hostIpAddress[1024];
    uint16_t ClientId;

    CHMultiforcerNetworkGeneral GeneralData;
    CHMultiforcerNeworkWorkunitRobust WorkunitData;
    CHWorkunitRobustElement Workunit;
};

class CHNetworkServerInstance {
public:

    CHNetworkServerInstance(boost::asio::io_service& io_service, short port, 
            struct classContainerStruct newClassContainer)
    : io_service_(io_service),
    acceptor_(io_service, tcp::endpoint(tcp::v4(), port)) {
        this->classContainer = newClassContainer;
        //printf("Server constructor\n");
        CHNetworkServerSession* new_session = new CHNetworkServerSession(io_service_, this->classContainer);
        acceptor_.async_accept(new_session->socket(),
                boost::bind(&CHNetworkServerInstance::handle_accept, this, new_session,
                boost::asio::placeholders::error));
    }

    void handle_accept(CHNetworkServerSession* new_session,
            const boost::system::error_code& error) {
        //printf("In server::handle_accept()\n");
        if (!error) {
            new_session->start();
            new_session = new CHNetworkServerSession(io_service_, this->classContainer);
            acceptor_.async_accept(new_session->socket(),
                    boost::bind(&CHNetworkServerInstance::handle_accept, this, new_session,
                    boost::asio::placeholders::error));
        } else {
            delete new_session;
        }
    }

private:
    boost::asio::io_service& io_service_;
    tcp::acceptor acceptor_;
    struct classContainerStruct classContainer;
};


#endif