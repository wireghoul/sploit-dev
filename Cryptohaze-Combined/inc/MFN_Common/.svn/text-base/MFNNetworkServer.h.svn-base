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

#ifndef __MFNNETWORKSERVER_H__
#define __MFNNETWORKSERVER_H__

#include "MFN_Common/MFNNetworkCommon.h"
#include "MFN_Common/MFNNetworkRPC.pb.h"
#include <CH_Common/CHHiresTimer.h>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <deque>

// Timeout timer interval in seconds
#define MFN_SERVER_TIMER_INTERVAL 45
// Time to consider a client dead.
#define MFN_SERVER_TIMEOUT_PERIOD 60

using boost::asio::ip::tcp;

class MFNNetworkServerInstance;
class MFNNetworkServer;


class MFNNetworkServer {
public:
    /**
     * Default constructor - requires the network port to use.
     *
     * @param port The network port to use.
     */
    MFNNetworkServer(uint16_t port);

    /**
     * Start the network server
     */
    virtual int startNetwork();
    /**
     * Terminate all network server threads
     */
    virtual int stopNetwork();

protected:
    // TCP port number that will be used for this connection.
    int portNumber;

    boost::asio::io_service io_service;
    MFNNetworkServerInstance *NetworkServer;

    boost::thread *ioThreads[MFN_MAX_IO_THREADS];
};

class MFNNetworkServerSession {
public:

    MFNNetworkServerSession(boost::asio::io_service& io_service)
    : socket_(io_service) {
        this->displayThreadId = 0;
        // Reserve 1MB for pending data.
        this->pendingData.reserve(1024*1024);
        this->timer = new boost::asio::deadline_timer(io_service);
        this->connectionIsAlive = 0;
    }
    
    ~MFNNetworkServerSession() {
        this->timer->cancel();
        //TODO: Figure out how to delete the timer without segfaults.
        // Smart pointer or such?
        //this->timer->wait();
        //delete this->timer;
    }

    tcp::socket& socket() {
        return socket_;
    }

    void start();

    void handle_read(const boost::system::error_code& error,
        size_t bytes_transferred);

    void handle_write(const boost::system::error_code& error);

private:
    tcp::socket socket_;
    enum {
        max_length = (1024*1024)
    };
    char data_[max_length];

    /**
     * The thread ID from the display class - used to send rates/etc to the 
     * correct thread.
     */
    int displayThreadId;
    /**
     * The remote host IP address.  This is stored, as the IP is unavailable
     * after the session is torn down, and the disconnect message should display
     * this for the user sanity.
     */
    std::string remoteHostIpAddress;
    
    uint32_t ClientId;

    // Protobufs for the communication - try to reuse these as much as possible.
    MFNRPCRequest RPCRequest;
    MFNRPCResponse RPCResponse;
    
    /**
     * Buffer for pending data.  If the read function has not read enough data
     * to service the request, it will issue another async read until this has
     * been filled.  Once it is filled with enough data, the data will be
     * parsed properly.  A vector is used as it can be addressed as linear
     * space.
     */
    std::vector<uint8_t> pendingData;
    
    /**
     * Timer for the connection.  This starts when the connection is opened and
     * runs until the connection is closed.  This enables the use of a timer to
     * detect a dead connection or dead client and close it.
     */
    CHHiresTimer totalConnectedTime;
    
    /**
     * The last time the client sent data.  This is used in combination with an
     * alarm to disconnect failed clients.  On any complete communication from
     * the client, this is updated with the current value of the
     * totalConnectedTime timer.
     */
    double lastClientUpdate;
    
    /**
     * Timeout timer.
     */
    boost::asio::deadline_timer *timer;

    /**
     * Set to true if the connection is open to prevent double-closing it.
     */
    char connectionIsAlive;
    
    /**
     * Enough data has been read - perform actions on the pending data.
     */
    void performActionsOnPendingData();
    
    /**
     * Respond with the general hash information.
     */
    void sendGeneralInformation();
    
    /**
     * Respond with the uncracked hash data from the hashfile class.
     */
    void sendUncrackedHashes();
    
    /**
     * Respond with the uncracked salts from the hashfile class.  Not currently
     * used...
     */
    void sendUncrackedSalts();
    
    /**
     * Respond with the current charset.
     */
    void sendCharset();
    
    /**
     * Respond with the requested workunits from the local workunit class.
     */
    void sendWorkunits();
    
    /**
     * Submit a workunit as completed.
     */
    void submitWorkunit();
    
    /**
     * Cancels a workunit
     */
    void cancelWorkunit();
    
    /**
     * Receives a found password
     */
    void submitPassword();
    
    /**
     * Deals with rate submission
     */
    void submitRate();
    
    /**
     * Update the last client data time for the timeout detection.
     */
    void updateLastClientDataTime() {
        this->lastClientUpdate = this->totalConnectedTime.getElapsedTime();
        this->timer->expires_from_now(
            boost::posix_time::seconds(MFN_SERVER_TIMER_INTERVAL));
        this->timer->async_wait(
            boost::bind(&MFNNetworkServerSession::timerTimeout, this, 
            boost::asio::placeholders::error));
    }
    
    /**
     * Handle the timer timeouts.  If enough elapsed time has passed since the
     * last checkin, kill the connection.
     * 
     * @param error Error code passed from async_wait
     */
    void timerTimeout(const boost::system::error_code& error);
    
    /**
     * Closes the connection and deletes the class.
     */
    void closeServerConnection();
    
    /**
     * Writes the RPCResponse to the network.
     */
    void sendRPCResponse();
};

class MFNNetworkServerInstance {
public:

    MFNNetworkServerInstance(boost::asio::io_service& io_service, short port)
    : io_service_(io_service),
    acceptor_(io_service, tcp::endpoint(tcp::v4(), port)) {
        MFNNetworkServerSession* new_session = new MFNNetworkServerSession(
                io_service_);
        acceptor_.async_accept(new_session->socket(),
                boost::bind(&MFNNetworkServerInstance::handle_accept, this, 
                new_session, boost::asio::placeholders::error));
    }

    void handle_accept(MFNNetworkServerSession* new_session,
            const boost::system::error_code& error) {
        //printf("In server::handle_accept()\n");
        if (!error) {
            new_session->start();
            new_session = new MFNNetworkServerSession(io_service_);
            acceptor_.async_accept(new_session->socket(),
                    boost::bind(&MFNNetworkServerInstance::handle_accept, this, new_session,
                    boost::asio::placeholders::error));
        } else {
            delete new_session;
        }
    }

private:
    boost::asio::io_service& io_service_;
    tcp::acceptor acceptor_;
};


#endif