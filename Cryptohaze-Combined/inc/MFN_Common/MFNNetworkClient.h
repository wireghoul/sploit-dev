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

#ifndef __MFNNETWORKCLIENT_H__
#define __MFNNETWORKCLIENT_H__

#include "MFN_Common/MFNNetworkCommon.h"
#include "MFN_Common/MFNNetworkRPC.pb.h"
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>

using boost::asio::ip::tcp;

class MFNNetworkClient {
public:
    /**
     * Attempts to connect to a remote host with the specified information.
     * 
     * This constructor starts the network client, and connect to the specified
     * remote host.  If oneshot is true, if it fails, it will exit and report
     * the failure.  If oneshot is false (default), it will continue trying to
     * connect to the remote host, sleeping inbetween connect attempts.  This 
     * is useful for "drone" systems - bring the client up, and when the server
     * runs, it will connect to the server, do the work, and then wait for the
     * server to come back up again.
     * 
     * @param hostname The hostname to connect to (IP, DNS, etc)
     * @param port The remote port to connect to
     * @param oneshot True if exit on failure, false if wait on failure.
     */
    MFNNetworkClient(std::string hostname, uint16_t port, char oneshot);

    /**
     * Updates the MFNGeneralInformation structure with the current information
     * from the server.  Will fetch the current class itsself.
     * 
     * @return 0 on success, error code on failure.
     */
    uint32_t updateGeneralInfo();

    /**
     * Updates the current hash file class with the uncracked hashes from the
     * server.  This will fetch the hashfile itsself.
     * 
     * @return 0 on success, error code on failure.
     */
    uint32_t updateUncrackedHashes();
    
    /**
     * Same as above...
     */
    uint32_t updateUncrackedSalts();
    
    uint32_t updateCharset();
    
    /**
     * Requests the specified number of workunits from the remote host.  This
     * can be used to prefetch units, and have some on hand immediately.  This
     * may or may not be immediately useful, but going forward it certainly
     * seems like it has some potential uses!
     * 
     * @param numberWorkunits The number of workunits to request.
     * @param passwordLength The current password length.  If this does not
     * match the remote system, it will return a null set.
     * @return 0 on success, errorcode on failure.
     */
    uint32_t fetchWorkunits(uint32_t numberWorkunits, uint32_t passwordLength);
    
    /**
     * Submits a found hash/password pair to the remote server. 
     * 
     * @param foundHash The hash string matching the password.
     * @param foundPassword A string containing the found password.
     * @param algorithmType The algorithm type identifier
     * @return 0 on success, errorcode on failure.
     */
    uint32_t submitFoundHash(std::vector<uint8_t> foundHash, 
        std::vector<uint8_t> foundPassword, uint8_t algorithmType);
    
    /**
     * Submits or cancels a completed workunit by ID to the server.
     * 
     * Once a workunit has been fully completed with all found hashes submitted,
     * this function will "close out" the workunit on the server.  If it is a
     * cancel request, it will cancel out on the server and requeue for further
     * execution by another resource. 
     * 
     * @param finishedWorkunitId The ID of the completed workunit
     * @return 0 on success, errorcode on failure.
     */
    uint32_t submitWorkunit(uint64_t finishedWorkunitId);
    uint32_t cancelWorkunit(uint64_t finishedWorkunitId);
    
    /**
     * Send just the system cracking rate to the server.
     * 
     * While the other functions send the system cracking rate to the server
     * with their submissions, this function simply sends the specified cracking
     * rate to the server.  This function does NOT pull from the display class,
     * but sends the parameter directly to the server.
     * 
     * @param crackingRate The cracking rate, in passwords/second.
     * @return 0 on success, errorcode on failure.
     */
    uint32_t submitSystemCrackingRate(uint64_t crackingRate);
    
protected:

    /**
     * Mutex to protect the network socket during an RPC request.  There may be
     * multiple threads using the network class, and the requests/responses
     * MUST NOT be mixed up in transit or things will just fail badly.  This
     * mutex is used to ensure that a single RPC request/response is completed
     * before anything else is allowed to use the socket.
     */
    boost::mutex rpcMutex;

    /**
     * RPC request and response holders.  As there is only one network socket,
     * there will only be one request/response inflight at any given time.  Any
     * access to these must own the rpcMutex before using them.
     */
    MFNRPCRequest RPCRequest;
    MFNRPCResponse RPCResponse;

    /**
     * Perform an RPC transaction with the private RPCRequest/RPCResponse
     * protobufs.  This function does NOT acquire the rpcMutex, and must be
     * called from a function that has already obtained it.
     */
    uint32_t makeRPCRequest();

    // Network related things
    boost::asio::io_service io_service;
    tcp::socket *socket;
    
};


#endif