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

#include "MFN_Common/MFNNetworkCommon.h"
#include "MFN_Common/MFNNetworkServer.h"

#include "MFN_Common/MFNMultiforcerClassFactory.h"
#include "MFN_Common/MFNGeneralInformation.h"

#include "CH_HashFiles/CHHashFileV.h"
#include "CH_Common/CHCharsetNew.h"
//#define TRACE_PRINTF 1

#include "MFN_Common/MFNDebugging.h"

#include "MFN_Common/MFNDisplay.h"
#include "MFN_Common/MFNWorkunitBase.h"

#include <boost/lexical_cast.hpp>

// Global class constructor factory
extern MFNClassFactory MultiforcerGlobalClassFactory;


void RunIoService(boost::asio::io_service* io_service_param) {
    trace_printf("RunIoService()\n");
    for (;;) {
    try
    {
      io_service_param->run();
      break; // run() exited normally
    }
    catch (boost::system::system_error& e)
    {
        network_printf("\n\nGOT EXCEPTION IN RunIoService!!!\n");
        network_printf("Exception data: %s\n", e.what());
        io_service_param->reset();

      // Deal with exception as appropriate.
    }
  }
}

MFNNetworkServer::MFNNetworkServer(uint16_t port) {
    trace_printf("MFNNetworkServer::MFNNetworkServer(%d)\n", port);
    // Set the network port as requested.
    this->portNumber = port;
    this->NetworkServer = NULL;
}


// Start up all the IO threads to handle traffic.
int MFNNetworkServer::startNetwork() {
    trace_printf("MFNNetworkServer::startNetwork()\n");
    int i;

    // Only create the network server if it's not already present.
    if (!this->NetworkServer) {
        this->NetworkServer = new MFNNetworkServerInstance(this->io_service, this->portNumber);
    }

    // Launch all the network IO threads
    for (i = 0; i < MFN_MAX_IO_THREADS; i++) {
        this->ioThreads[i] = new boost::thread(RunIoService, &this->io_service);
    }
    return 1;
}

// Stop the io_service instances, and wait for all the threads to return.
int MFNNetworkServer::stopNetwork() {
    trace_printf("MFNNetworkServer::stopNetwork()\n");
    int i;

    this->io_service.stop();
    for (i = 0; i < MFN_MAX_IO_THREADS; i++) {
        this->ioThreads[i]->join();
    }
    return 1;
}




// MFNNetworkServerSession functions
void MFNNetworkServerSession::start() {
    trace_printf("MFNNetworkServerSession::start()\n");
    MFNDisplay *Display;
    std::string DisplayUpdate;
    
    Display = MultiforcerGlobalClassFactory.getDisplayClass();

    // Get the thread ID for the display
    this->displayThreadId = Display->getFreeThreadId(NETWORK_HOST);

    // Store the IP address.
    this->remoteHostIpAddress = socket_.remote_endpoint().address().to_string();
    
    // Start the session timer and set up the timeout.
    this->totalConnectedTime.start();
    this->updateLastClientDataTime();
    
    // Mark the connection as alive.
    this->connectionIsAlive = 1;

    DisplayUpdate = boost::lexical_cast<std::string>(this->displayThreadId) +
            ": " + this->remoteHostIpAddress;
    Display->addStatusLine(DisplayUpdate);
    Display->alterNetworkClientCount(1);

    this->ClientId = MultiforcerGlobalClassFactory.getWorkunitClass()->GetClientId();

    // Set the thread speed to 0 for now.
    Display->setThreadCrackSpeed(this->displayThreadId, 0.00);

    socket_.async_read_some(boost::asio::buffer(data_, max_length),
            boost::bind(&MFNNetworkServerSession::handle_read, this,
            boost::asio::placeholders::error,
            boost::asio::placeholders::bytes_transferred));
}



void MFNNetworkServerSession::handle_read(const boost::system::error_code& error,
    size_t bytes_transferred) {
    trace_printf("MFNNetworkServerSession::handle_read()\n");


    // If there is no error, read the RPC buffer and perform the action.
    // If there IS an error, the client has disconnected - perform the needed
    // actions.
    if (!error) {
        if (bytes_transferred > 0) {
            // Update the client last seen time - we have data!
            this->updateLastClientDataTime();
            
            // Do what the command requests.
            network_printf("handle_read read %d bytes\n", bytes_transferred);
            
            network_printf("Data: ");
            for (int i = 0; i < bytes_transferred; i++) {
                network_printf("%02x ", this->data_[i]);
            }
            network_printf("\n\n");
            
            // Copy the data to the end of the current vector.
            // If the vector is too small to fit the data, resize it.
            if (this->pendingData.capacity() < (this->pendingData.size() + bytes_transferred)) {
                network_printf("pendingData capacity too small!\n");
                network_printf("pendingData preCapacity: %d\n", this->pendingData.capacity());
                this->pendingData.reserve(this->pendingData.size() + bytes_transferred);
                network_printf("pendingData postCapacity: %d\n", this->pendingData.capacity());
            }
            // Store the old vector size so we can copy to the right place.
            size_t pendingSize = this->pendingData.size();
            network_printf("old pending size: %d\n", pendingSize);
            // Expand the vector to the new size.
            this->pendingData.resize(pendingSize + bytes_transferred, 0);
            network_printf("Buffer resized to %d\n", this->pendingData.size());
            memcpy(&this->pendingData[pendingSize], &this->data_[0], bytes_transferred);
            
            // If at least 4 bytes have come in, check.
            if (this->pendingData.size() >= 4) {
                uint32_t messageSize;
                messageSize = *(uint32_t *)&this->pendingData[0];
                network_printf("Message size: %u bytes\n", *(uint32_t *)&this->data_[0]);
                // Determine if the specified number of bytes is ready.  If so,
                // perform the actions on the data (which will remove this from
                // the buffer).  If not, just wait for more data.
                if (this->pendingData.size() >= (messageSize + 4)) {
                    this->performActionsOnPendingData();
                }
                
                // In any case, we probably want to read more now.
                this->socket_.async_read_some(boost::asio::buffer(data_, max_length),
                    boost::bind(&MFNNetworkServerSession::handle_read, this,
                    boost::asio::placeholders::error,
                    boost::asio::placeholders::bytes_transferred));
                
            }
        }
    } else {
        network_printf("Closing connection from handle_read\n");
        //std::cout << error.message() << std::endl;
        // Delete the connection if it's alive.  Otherwise it's already gone.
        if (this->connectionIsAlive) {
            this->closeServerConnection();
        }
        network_printf("Deleting this\n");
        delete this;
    }
}

void MFNNetworkServerSession::performActionsOnPendingData() {
    trace_printf("MFNNetworkServerSession::performActionsOnPendingData()\n");
    network_printf("Perform pending actions!\n");
    
    // Enough data has been collected to justify us doing some work on it.
    uint32_t messageSize;
    std::string messageData;
    
    network_printf("pending data size: %d\n", this->pendingData.size());
    
    messageSize = *(uint32_t *)&this->pendingData[0];
    network_printf("Pending actions: message size %d\n", messageSize);
    
    // Create the string with the message - skip the first 4 bytes (size).
    messageData = std::string((const char *)&this->pendingData[4], messageSize);
    
    network_printf("Got message string of length %d\n", messageData.size());

    network_printf("Request string: ");
    for (int i = 0; i < messageData.length(); i++) {
        network_printf("%02x ", messageData[i]);
    }
    network_printf("\n\n");
    
    this->RPCRequest.Clear();
    if (!this->RPCRequest.ParseFromString(messageData)) {
        network_printf("Error parsing RPC request!\n");
    }
    //network_printf("Got RPC: %s\n", this->RPCRequest.DebugString().c_str());
    
    // TODO: Ensure the protobuf parsed properly and that the version is correct
    
    if (this->RPCRequest.has_system_cracking_rate()) {
        network_printf("RPC has cracking rate - submitting!\n");
        MultiforcerGlobalClassFactory.getDisplayClass()->
                setThreadCrackSpeed(this->displayThreadId, 
                (float)this->RPCRequest.system_cracking_rate());
    }
    
    // Switch based on the request
    switch(this->RPCRequest.request_id()) {
        case RPC_REQUEST_GENERAL_INFO:
            network_printf("Got request for general info!\n");
            this->sendGeneralInformation();
            break;
        case RPC_REQUEST_UNCRACKED_HASHES:
            network_printf("Got request for uncracked hashes!\n");
            this->sendUncrackedHashes();
            break;
        case RPC_REQUEST_UNCRACKED_SALTS:
            network_printf("Got request for uncracked salts!\n");
            this->sendUncrackedSalts();
            break;
        case RPC_REQUEST_CHARSET:
            network_printf("Got request for charset!\n");
            this->sendCharset();
            break;
        case RPC_REQUEST_GET_WORKUNITS:
            network_printf("Got request for workunits!\n");
            this->sendWorkunits();
            break;
        case RPC_REQUEST_SUBMIT_WORKUNITS:
            network_printf("Got request for submit workunits!\n");
            this->submitWorkunit();
            break;
        case RPC_REQUEST_CANCEL_WORKUNITS:
            network_printf("Got request for cancel workunits!\n");
            this->cancelWorkunit();
            break;
        case RPC_REQUEST_SUBMIT_PASSWORD:
            network_printf("Got request for submit password!\n");
            this->submitPassword();
            break;
        case RPC_REQUEST_SUBMIT_RATE:
            network_printf("Got request for submit rate!\n");
            this->submitRate();
            break;
        default:
            network_printf("Unknown request!\n");
            // Send a "function not implemented" response.
            this->RPCResponse.Clear();
            this->RPCResponse.set_system_version_id(MFN_NETWORK_SERVER_VERSION);
            this->RPCResponse.set_response_type_id(RPC_ERROR_FUNCTION_NOT_IMPLEMENTED);
            this->sendRPCResponse();
            break;
    }
    
    // Remove the data from the beginning of the vector once it has been read.
    this->pendingData.erase(this->pendingData.begin(), 
            this->pendingData.begin() + (messageSize + 4));
}

void MFNNetworkServerSession::sendGeneralInformation() {
    trace_printf("MFNNetworkServerSession::sendGeneralInformation()\n");
    MFNGeneralInformation *GeneralInformation;
    
    GeneralInformation = MultiforcerGlobalClassFactory.
            getGeneralInformationClass();

    this->RPCResponse.Clear();
    this->RPCResponse.set_system_version_id(MFN_NETWORK_SERVER_VERSION);
    this->RPCResponse.set_response_type_id(RPC_RESPONSE_GENERAL_INFO);
    this->RPCResponse.set_hash_file_id(GeneralInformation->getHashId());
    this->RPCResponse.set_hash_type_id(GeneralInformation->getHashId());
    this->RPCResponse.set_charset_id(GeneralInformation->getCharsetClassId());
    this->RPCResponse.set_password_length(GeneralInformation->getPasswordLength());
    this->RPCResponse.set_system_version_id(MFN_NETWORK_SERVER_VERSION);
    
    this->sendRPCResponse();
}

void MFNNetworkServerSession::sendUncrackedHashes() {
    trace_printf("MFNNetworkServerSession::sendUncrackedHashes()\n");
    std::string uncrackedHashesProtobuf;
    
    // Get the protobuf out of the hashfile class.
    MultiforcerGlobalClassFactory.getHashfileClass()->
        ExportHashListToRemoteSystem(&uncrackedHashesProtobuf);
    
    // Set up the response, to include the uncracked hash data.
    this->RPCResponse.Clear();
    this->RPCResponse.set_system_version_id(MFN_NETWORK_SERVER_VERSION);
    this->RPCResponse.set_response_type_id(RPC_RESPONSE_UNCRACKED_HASHES);
    this->RPCResponse.set_additional_data(uncrackedHashesProtobuf);
    
    this->sendRPCResponse();
}

void MFNNetworkServerSession::sendUncrackedSalts() {
    trace_printf("MFNNetworkServerSession::sendUncrackedSalts()\n");
    std::string uncrackedSaltsProtobuf;
    
    // Get the protobuf out of the hashfile class.
    MultiforcerGlobalClassFactory.getHashfileClass()->
        ExportUniqueSaltsToRemoteSystem(&uncrackedSaltsProtobuf);
    
    // Set up the response, to include the uncracked hash data.
    this->RPCResponse.Clear();
    this->RPCResponse.set_system_version_id(MFN_NETWORK_SERVER_VERSION);
    this->RPCResponse.set_response_type_id(RPC_RESPONSE_UNCRACKED_SALTS);
    this->RPCResponse.set_additional_data(uncrackedSaltsProtobuf);
    
    this->sendRPCResponse();
}

void MFNNetworkServerSession::sendCharset() {
    trace_printf("MFNNetworkServerSession::sendCharset()\n");
    std::string charsetProtobuf;
    
    // Get the protobuf out of the hashfile class.
    MultiforcerGlobalClassFactory.getCharsetClass()->
        ExportCharsetToRemoteSystem(&charsetProtobuf);
    
    // Set up the response, to include the uncracked hash data.
    this->RPCResponse.Clear();
    this->RPCResponse.set_system_version_id(MFN_NETWORK_SERVER_VERSION);
    this->RPCResponse.set_response_type_id(RPC_RESPONSE_CHARSET);
    this->RPCResponse.set_additional_data(charsetProtobuf);
    
    this->sendRPCResponse();
}

void MFNNetworkServerSession::sendWorkunits() {
    trace_printf("MFNNetworkServerSession::sendWorkunits()\n");
    std::string workunitProtobuf;
    
    // Get the protobuf with the requested number of workunits.
    MultiforcerGlobalClassFactory.getWorkunitClass()->
        ExportWorkunitsAsProtobuf(this->RPCRequest.number_workunits_requested(),
        this->ClientId, &workunitProtobuf, this->RPCRequest.password_length());
    
    // Set up the response, to include the uncracked hash data.
    this->RPCResponse.Clear();
    this->RPCResponse.set_system_version_id(MFN_NETWORK_SERVER_VERSION);
    this->RPCResponse.set_response_type_id(RPC_RESPONSE_GET_WORKUNITS);
    this->RPCResponse.set_additional_data(workunitProtobuf);
    
    this->sendRPCResponse();
}

void MFNNetworkServerSession::submitWorkunit() {
    trace_printf("MFNNetworkServerSession::submitWorkunit()\n");
    // Submit the workunit ID as completed.
    MultiforcerGlobalClassFactory.getWorkunitClass()->
            SubmitWorkunitById(this->RPCRequest.submitted_workunit_id());
    
    // Set up the response, to include the uncracked hash data.
    this->RPCResponse.Clear();
    this->RPCResponse.set_system_version_id(MFN_NETWORK_SERVER_VERSION);
    this->RPCResponse.set_response_type_id(RPC_RESPONSE_SUBMIT_WORKUNITS);
    
    this->sendRPCResponse();
}

void MFNNetworkServerSession::cancelWorkunit() {
    trace_printf("MFNNetworkServerSession::cancelWorkunit()\n");
    // Submit the workunit ID as cancelled.
    MultiforcerGlobalClassFactory.getWorkunitClass()->
            CancelWorkunitById(this->RPCRequest.submitted_workunit_id());
    
    this->RPCResponse.Clear();
    this->RPCResponse.set_system_version_id(MFN_NETWORK_SERVER_VERSION);
    this->RPCResponse.set_response_type_id(RPC_RESPONSE_CANCEL_WORKUNITS);
    
    this->sendRPCResponse();
}

void MFNNetworkServerSession::submitPassword() {
    trace_printf("MFNNetworkServerSession::submitPassword()\n");
    // Submit the hash to the hashfile class
    std::vector<uint8_t> hashValue = 
            std::vector<uint8_t>(this->RPCRequest.found_password_hash().begin(), 
            this->RPCRequest.found_password_hash().end());
    std::vector<uint8_t> passwordValue = 
            std::vector<uint8_t>(this->RPCRequest.found_password_value().begin(), 
            this->RPCRequest.found_password_value().end());
    uint8_t algorithmType = (uint8_t)this->RPCRequest.algorithm_type();
    
    MultiforcerGlobalClassFactory.getHashfileClass()->
        ReportFoundPassword(hashValue, passwordValue, algorithmType);
    MultiforcerGlobalClassFactory.getDisplayClass()->addCrackedPassword(passwordValue);
    
    this->RPCResponse.Clear();
    this->RPCResponse.set_system_version_id(MFN_NETWORK_SERVER_VERSION);
    this->RPCResponse.set_response_type_id(RPC_RESPONSE_SUBMIT_PASSWORD);
    
    this->sendRPCResponse();
}

void MFNNetworkServerSession::submitRate() {
    trace_printf("MFNNetworkServerSession::submitRate()\n");
    // Rate is processed if it's present in any protobuf.  Do nothing except
    // ack the receipt.
    this->RPCResponse.Clear();
    this->RPCResponse.set_system_version_id(MFN_NETWORK_SERVER_VERSION);
    this->RPCResponse.set_response_type_id(RPC_RESPONSE_SUBMIT_RATE);
    
    this->sendRPCResponse();
}

void MFNNetworkServerSession::sendRPCResponse() {
    trace_printf("MFNNetworkServerSession::sendRPCResponse()\n");
    
    std::string responseString = this->RPCResponse.SerializeAsString();
    
    uint32_t responseSize = responseString.length();
        
    boost::system::error_code error = boost::asio::error::access_denied;

    // Write the data length in bytes out.
    boost::asio::write(this->socket_,
            boost::asio::buffer(&responseSize, sizeof(responseSize)),
            boost::asio::transfer_all(), error);
    // Write the actual data in bytes.
    boost::asio::write(this->socket_,
            boost::asio::buffer(responseString),
            boost::asio::transfer_all(), error);
}

void MFNNetworkServerSession::handle_write(const boost::system::error_code& error) {
    trace_printf("MFNNetworkServerSession::handle_write()\n");
    if (!error) {
        socket_.async_read_some(boost::asio::buffer(data_, max_length),
                boost::bind(&MFNNetworkServerSession::handle_read, this,
                boost::asio::placeholders::error,
                boost::asio::placeholders::bytes_transferred));
    } else {
        network_printf("Closing connection from handle_write\n");
        if (this->connectionIsAlive) {
            this->closeServerConnection();
        }
    }
}

void MFNNetworkServerSession::closeServerConnection() {
    trace_printf("MFNNetworkServerSession::closeServerConnection()\n");
    network_printf("Closing connection.\n");
    // Report the disconnect
    std::string DisconnectString;
    MFNDisplay *Display;
    MFNWorkunitBase *Workunit;

    // Only do this if the connection is still alive.  Don't double-close it.
    if (this->connectionIsAlive) {
        network_printf("Actually closing connection\n");
        Display = MultiforcerGlobalClassFactory.getDisplayClass();
        Workunit = MultiforcerGlobalClassFactory.getWorkunitClass();

        DisconnectString = "DSC: " + this->remoteHostIpAddress;

        Display->addStatusLine(DisconnectString);
        Display->alterNetworkClientCount(-1);
        Display->releaseThreadId(this->displayThreadId);

        Workunit->CancelAllWorkunitsByClientId(this->ClientId);
        Workunit->FreeClientId(this->ClientId);

        this->connectionIsAlive = 0;
        
        // Only close the socket if it is open.
        if (this->socket_.is_open()) {
            this->socket_.close();
        }
        this->timer->cancel();
    } else {
        network_printf("Connection already closed. skipping.\n");
    }
}

void MFNNetworkServerSession::timerTimeout(const boost::system::error_code& error) {
    trace_printf("MFNNetworkServerSession::timerTimeout()\n");
    network_printf("In timer timeout.\n");
    double timeSinceLastCheckin;

    // If cancel() is called, this function will get called - do nothing.
    if (error == boost::asio::error::operation_aborted) {
        network_printf("Timer was canceled\n");
        return;
    }
    
    // Calculate the time since last data seen.
    timeSinceLastCheckin = this->totalConnectedTime.getElapsedTime() -
        this->lastClientUpdate;
    
    // If it's been too long, kill the connection.
    if (timeSinceLastCheckin > MFN_SERVER_TIMEOUT_PERIOD) {
        network_printf("Timeout exceeded.\n");
        network_printf("Closing connection from timer\n");
        this->closeServerConnection();
        return;
    }
    
    // Reset the timer and wait.
    this->timer->expires_from_now(
        boost::posix_time::seconds(MFN_SERVER_TIMER_INTERVAL));
    this->timer->async_wait(
        boost::bind(&MFNNetworkServerSession::timerTimeout, this, 
        boost::asio::placeholders::error));
}

//#define UNIT_TEST 1
#if UNIT_TEST
#include <stdlib.h>
#include <stdio.h>

int main() {
    MFNNetworkServer *Server;

    Server = new MFNNetworkServer(12345);

    Server->startNetwork();

    sleep(60);
    Server->stopNetwork();
    
}

#endif