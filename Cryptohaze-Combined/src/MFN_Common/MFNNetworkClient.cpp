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

#include "MFN_Common/MFNNetworkClient.h"
#include "MFN_Common/MFNMultiforcerClassFactory.h"
#include "MFN_Common/MFNGeneralInformation.h"
#include "CH_Common/CHCharsetNew.h"
#include "CH_HashFiles/CHHashFileV.h"
#include "MFN_Common/MFNWorkunitBase.h"
#include "MFN_Common/MFNDisplay.h"
#include "Multiforcer_Common/CHCommon.h"
#include "MFN_Common/MFNDebugging.h"

extern MFNClassFactory MultiforcerGlobalClassFactory;
extern struct global_commands global_interface;

// Create a new network client to the given hostname.
MFNNetworkClient::MFNNetworkClient(std::string hostname, uint16_t port, char oneshot) {
    trace_printf("MFNNetworkClient::MFNNetworkClient()\n");
    char waitMessagePrinted = 0;
    // Keep trying until this works or is terminated with ctrl-c
    while(!global_interface.user_exit) {
        try {
            // Look up the hostname
            char portBuffer[16];
            sprintf(portBuffer, "%d", port);
            tcp::resolver resolver(this->io_service);
            tcp::resolver::query query(hostname.c_str(), portBuffer);
            tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
            tcp::resolver::iterator end;

            this->socket = new tcp::socket(io_service);

            boost::system::error_code error = boost::asio::error::host_not_found;
            while (error && endpoint_iterator != end)
            {
              this->socket->close();
              this->socket->connect(*endpoint_iterator++, error);
            }
            
            if (error) {
                throw boost::system::system_error(error); // Some other error.
            }
            return;
        }
        catch (std::exception& e)
        {
            if (!waitMessagePrinted) {
                // This one exits directly, as it's an error in setup.
                std::cerr << "Network error: " << e.what() << std::endl;
                std::cerr << "Waiting for server... (press ctrl-c to exit)" << std::endl;
                std::cerr << "Server hostname: " << hostname << std::endl;
                waitMessagePrinted = 1;
            }
            CHSleep(5);
        }
    }
}


// Updates the general information from the server.
uint32_t MFNNetworkClient::updateGeneralInfo() {
    trace_printf("MFNNetworkClient::updateGeneralInfo()\n");
    this->rpcMutex.lock();
    uint32_t RPCResponseCode;
    this->RPCRequest.Clear();
    this->RPCRequest.set_request_id(RPC_REQUEST_GENERAL_INFO);

    RPCResponseCode = this->makeRPCRequest();

    // Ensure the response is of the type we aim for.
    if (RPCResponseCode != RPC_RESPONSE_GENERAL_INFO) {
        // Error - not what we wanted.  Return the error code.
        this->rpcMutex.unlock();
        return RPCResponseCode;
    } else {
        // General information is done.  Load it into the info class.
        MFNGeneralInformation *GeneralInformation;
        GeneralInformation = MultiforcerGlobalClassFactory.
                getGeneralInformationClass();
        GeneralInformation->setHashId(this->RPCResponse.hash_type_id());
        GeneralInformation->setCharsetClassId(this->RPCResponse.charset_id());
        GeneralInformation->setPasswordLength(this->RPCResponse.password_length());
        this->RPCResponse.Clear();
        this->rpcMutex.unlock();
        return 0;
    }
}


uint32_t MFNNetworkClient::updateUncrackedHashes() {
    trace_printf("MFNNetworkClient::updateUncrackedHashes()\n");
    this->rpcMutex.lock();
    
    uint32_t RPCResponseCode;
    this->RPCRequest.Clear();
    this->RPCRequest.set_request_id(RPC_REQUEST_UNCRACKED_HASHES);

    RPCResponseCode = this->makeRPCRequest();

    // Ensure the response is of the type we aim for.
    if (RPCResponseCode != RPC_RESPONSE_UNCRACKED_HASHES) {
        // Error - not what we wanted.  Return the error code.
        this->rpcMutex.unlock();
        return RPCResponseCode;
    } else {
        // General information is done.  Load it into the info class.
        CHHashFileV *HashFile;
        std::string Hashes;
        HashFile = MultiforcerGlobalClassFactory.getHashfileClass();
        Hashes = this->RPCResponse.additional_data();
        HashFile->ImportHashListFromRemoteSystem(Hashes);
        this->RPCResponse.Clear();
        this->rpcMutex.unlock();
        return 0;
    }
}

uint32_t MFNNetworkClient::updateUncrackedSalts() {
    trace_printf("MFNNetworkClient::updateUncrackedSalts()\n");
    this->rpcMutex.lock();
    
    uint32_t RPCResponseCode;
    this->RPCRequest.Clear();
    this->RPCRequest.set_request_id(RPC_REQUEST_UNCRACKED_SALTS);

    RPCResponseCode = this->makeRPCRequest();

    // Ensure the response is of the type we aim for.
    if (RPCResponseCode != RPC_RESPONSE_UNCRACKED_SALTS) {
        // Error - not what we wanted.  Return the error code.
        this->rpcMutex.unlock();
        return RPCResponseCode;
    } else {
        // General information is done.  Load it into the info class.
        CHHashFileV *HashFile;
        std::string Salts;
        HashFile = MultiforcerGlobalClassFactory.getHashfileClass();
        Salts = this->RPCResponse.additional_data();
        HashFile->ImportUniqueSaltsFromRemoteSystem(Salts);
        this->RPCResponse.Clear();
        this->rpcMutex.unlock();
        return 0;
    }
}

uint32_t MFNNetworkClient::updateCharset() {
    trace_printf("MFNNetworkClient::updateCharset()\n");
    this->rpcMutex.lock();
    
    uint32_t RPCResponseCode;
    this->RPCRequest.Clear();
    this->RPCRequest.set_request_id(RPC_REQUEST_CHARSET);

    RPCResponseCode = this->makeRPCRequest();

    // Ensure the response is of the type we aim for.
    if (RPCResponseCode != RPC_RESPONSE_CHARSET) {
        // Error - not what we wanted.  Return the error code.
        this->rpcMutex.unlock();
        return RPCResponseCode;
    } else {
        // General information is done.  Load it into the info class.
        CHCharsetNew *Charset;
        std::string CharsetProtobuf;
        Charset = MultiforcerGlobalClassFactory.getCharsetClass();
        CharsetProtobuf = this->RPCResponse.additional_data();
        Charset->ImportCharsetFromRemoteSystem(CharsetProtobuf);
        this->RPCResponse.Clear();
        this->rpcMutex.unlock();
        return 0;
    }
}

uint32_t MFNNetworkClient::submitFoundHash(std::vector<uint8_t> foundHash, 
        std::vector<uint8_t> foundPassword, uint8_t algorithmType) {
    trace_printf("MFNNetworkClient::submitFoundHash()\n");
    this->rpcMutex.lock();
    
    uint32_t RPCResponseCode;
    this->RPCRequest.Clear();
    this->RPCRequest.set_request_id(RPC_REQUEST_SUBMIT_PASSWORD);
    this->RPCRequest.set_found_password_hash(&foundHash[0], foundHash.size());
    this->RPCRequest.set_found_password_value(&foundPassword[0], foundPassword.size());
    this->RPCRequest.set_algorithm_type(algorithmType);
    this->RPCRequest.set_system_cracking_rate(
            (uint64_t)MultiforcerGlobalClassFactory.getDisplayClass()->
            getCurrentCrackRate());

    RPCResponseCode = this->makeRPCRequest();

    // Ensure the response is of the type we aim for.
    if (RPCResponseCode != RPC_RESPONSE_SUBMIT_PASSWORD) {
        // Error - not what we wanted.  Return the error code.
        this->rpcMutex.unlock();
        return RPCResponseCode;
    } else {
        this->RPCResponse.Clear();
        this->rpcMutex.unlock();
        return 0;
    }
}

uint32_t MFNNetworkClient::submitSystemCrackingRate(uint64_t crackingRate) {
    trace_printf("MFNNetworkClient::submitSystemCrackingRate()\n");
    this->rpcMutex.lock();
    
    uint32_t RPCResponseCode;
    this->RPCRequest.Clear();
    this->RPCRequest.set_request_id(RPC_REQUEST_SUBMIT_RATE);
    this->RPCRequest.set_system_cracking_rate(crackingRate);

    RPCResponseCode = this->makeRPCRequest();

    // Ensure the response is of the type we aim for.
    if (RPCResponseCode != RPC_RESPONSE_SUBMIT_RATE) {
        // Error - not what we wanted.  Return the error code.
        this->rpcMutex.unlock();
        return RPCResponseCode;
    } else {
        this->RPCResponse.Clear();
        this->rpcMutex.unlock();
        return 0;
    }
}

uint32_t MFNNetworkClient::fetchWorkunits(uint32_t numberWorkunits, 
        uint32_t passwordLength) {
    trace_printf("MFNNetworkClient::fetchWorkunits()\n");
    MFNWorkunitBase *Workunit = MultiforcerGlobalClassFactory.getWorkunitClass();

    this->rpcMutex.lock();
    
    uint32_t RPCResponseCode;
    this->RPCRequest.Clear();
    this->RPCRequest.set_request_id(RPC_REQUEST_GET_WORKUNITS);
    this->RPCRequest.set_number_workunits_requested(numberWorkunits);
    this->RPCRequest.set_password_length(passwordLength);
    this->RPCRequest.set_system_cracking_rate(
            (uint64_t)MultiforcerGlobalClassFactory.getDisplayClass()->
            getCurrentCrackRate());

    RPCResponseCode = this->makeRPCRequest();

    // Ensure the response is of the type we aim for.
    if (RPCResponseCode != RPC_RESPONSE_GET_WORKUNITS) {
        // Error - not what we wanted.  Return the error code.
        this->rpcMutex.unlock();
        return RPCResponseCode;
    } else {
        std::string Workunits;
        Workunits = this->RPCResponse.additional_data();
        Workunit->ImportWorkunitsFromProtobuf(Workunits);
        this->RPCResponse.Clear();
        this->rpcMutex.unlock();
        return 0;
    }
}

uint32_t MFNNetworkClient::submitWorkunit(uint64_t finishedWorkunitId) {
    trace_printf("MFNNetworkClient::submitWorkunit()\n");
    this->rpcMutex.lock();
    
    uint32_t RPCResponseCode;
    this->RPCRequest.Clear();
    this->RPCRequest.set_request_id(RPC_REQUEST_SUBMIT_WORKUNITS);
    this->RPCRequest.set_submitted_workunit_id(finishedWorkunitId);
    this->RPCRequest.set_system_cracking_rate(
            (uint64_t)MultiforcerGlobalClassFactory.getDisplayClass()->
            getCurrentCrackRate());

    RPCResponseCode = this->makeRPCRequest();

    // Ensure the response is of the type we aim for.
    if (RPCResponseCode != RPC_RESPONSE_SUBMIT_WORKUNITS) {
        // Error - not what we wanted.  Return the error code.
        this->rpcMutex.unlock();
        return RPCResponseCode;
    } else {
        this->RPCResponse.Clear();
        this->rpcMutex.unlock();
        return 0;
    }
}

uint32_t MFNNetworkClient::cancelWorkunit(uint64_t finishedWorkunitId) {
    trace_printf("MFNNetworkClient::cancelWorkunit()\n");
    this->rpcMutex.lock();
    
    uint32_t RPCResponseCode;
    this->RPCRequest.Clear();
    this->RPCRequest.set_request_id(RPC_REQUEST_CANCEL_WORKUNITS);
    this->RPCRequest.set_submitted_workunit_id(finishedWorkunitId);
    this->RPCRequest.set_system_cracking_rate(
            (uint64_t)MultiforcerGlobalClassFactory.getDisplayClass()->
            getCurrentCrackRate());

    RPCResponseCode = this->makeRPCRequest();

    // Ensure the response is of the type we aim for.
    if (RPCResponseCode != RPC_RESPONSE_CANCEL_WORKUNITS) {
        // Error - not what we wanted.  Return the error code.
        this->rpcMutex.unlock();
        return RPCResponseCode;
    } else {
        this->RPCResponse.Clear();
        this->rpcMutex.unlock();
        return 0;
    }
}


uint32_t MFNNetworkClient::makeRPCRequest() {
    trace_printf("MFNNetworkClient::makeRPCRequest()\n");
    // The outgoing request size, in bytes.
    uint32_t requestSize;
    // The outgoing request, as a string.
    std::string requestString;
    boost::system::error_code error = boost::asio::error::access_denied;
    
    try {
        requestString = this->RPCRequest.SerializeAsString();
        requestSize = (uint32_t)requestString.length();

        network_printf("Request string: ");
        for (int i = 0; i < requestString.length(); i++) {
            network_printf("%02x ", requestString[i]);
        }
        network_printf("\n\n");

        //network_printf("Protobuf: %s\n", this->RPCRequest.DebugString().c_str());

        // Write the data length in bytes out.
        boost::asio::write(*this->socket,
                boost::asio::buffer(&requestSize, sizeof(requestSize)),
                boost::asio::transfer_all(), error);
        // Write the actual data in bytes.
        boost::asio::write(*this->socket,
                boost::asio::buffer(requestString),
                boost::asio::transfer_all(), error);

        // Get the response from the server.
        uint32_t responseSize;
        // Response buffer is a byte array for use with boost::buffer
        uint8_t *responseBuffer;

        boost::asio::read(*this->socket, boost::asio::buffer(&responseSize, sizeof(responseSize)));
        network_printf("Got RPC response length of %d\n", responseSize);

        responseBuffer = new uint8_t[responseSize];
        boost::asio::read(*this->socket, boost::asio::buffer(responseBuffer, responseSize));
        this->RPCResponse.Clear();
        this->RPCResponse.ParseFromArray(responseBuffer, responseSize);
        network_printf("Protobuf: %s\n", this->RPCResponse.DebugString().c_str());
        return this->RPCResponse.response_type_id();
        delete[] responseBuffer;
    }
    catch (std::exception& e)
    {
        network_printf("makeRPCRequest network error: %s\n", e.what());
        //exit(1);
        global_interface.exit = 1;
        sprintf(global_interface.exit_message, "Network error: %s", e.what());
        return RPC_ERROR_SERVER_DISCONNECT;
    }
}


//#define UNIT_TEST 1
#ifdef UNIT_TEST
MFNClassFactory MultiforcerGlobalClassFactory;

int main() {
    MFNNetworkClient *Client;

    Client = new MFNNetworkClient(std::string("127.0.0.1"), 12410, 1);
    
    network_printf("Back from client.\n");
    sleep(5);

}

#endif