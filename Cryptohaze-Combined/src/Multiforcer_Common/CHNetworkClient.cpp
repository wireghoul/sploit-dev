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

#include "Multiforcer_Common/CHNetworkClient.h"

void printGeneralInfo(CHMultiforcerNetworkGeneral GeneralInfo) {
    printf("Version: %u\n", GeneralInfo.structure.version);
    printf("Hash ID: %u\n", GeneralInfo.structure.hash_id);
    printf("Password Length: %u\n", GeneralInfo.structure.password_length);
    printf("WU size (bits): %u\n", GeneralInfo.structure.workunit_size_bits);
    printf("Number hashes: %u\n", GeneralInfo.structure.number_hashes);
    printf("Charset Max: %u\n", GeneralInfo.structure.charset_max_characters);
    printf("Charset Length: %u\n", GeneralInfo.structure.charset_max_length);
}

// Commands to use for requesting things.
char generalInfoCommand[] = "g";
char hashListCommand[] = "h";
char charsetCommand[] = "c";
char workunitCommand[] = "w";
char submitWorkunitCommand[] = "s";
char submitPasswordCommand[] = "p";
char cancelWorkunitCommand[] = "l";

char newlineCommand[] = "\n";

extern struct global_commands global_interface;


// Create a new network client to the given hostname.
CHNetworkClient::CHNetworkClient(char *hostname, uint16_t port) {
    char waitMessagePrinted = 0;
    this->CurrentPasswordLength = 0;
    // Keep trying until this works or is terminated with ctrl-c
    while(1) {
        try {
            // Look up the hostname
            char portBuffer[16];
            sprintf(portBuffer, "%d", port);
            tcp::resolver resolver(this->io_service);
            tcp::resolver::query query(hostname, portBuffer);
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
              throw boost::system::system_error(error);
            }

            this->updateGeneralInfo();
            if (this->GeneralInfo.structure.version != NETWORK_SERVER_VERSION) {
                printf("Error!  Invalid server version: Need %d, got %d\n",
                        NETWORK_SERVER_VERSION, this->GeneralInfo.structure.version);
                exit(1);
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
                waitMessagePrinted = 1;
            }
            CHSleep(NETWORK_WAIT_TIME);
            if (global_interface.user_exit) {
                exit(1);
            }
        }
    }
}


// Updates the general information from the server.
void CHNetworkClient::updateGeneralInfo() {
    try {
        boost::system::error_code error = boost::asio::error::host_not_found;

        boost::asio::write(*this->socket, boost::asio::buffer(generalInfoCommand, strlen(generalInfoCommand)),
            boost::asio::transfer_all(), error);

        boost::asio::read(*this->socket, boost::asio::buffer(this->GeneralInfo.readbuffer));
        //printGeneralInfo(this->GeneralInfo);
        this->CurrentPasswordLength = this->GeneralInfo.structure.password_length;
        
        if (error) {
            throw boost::system::system_error(error); // Some other error.
        }
    }
    catch (std::exception& e)
    {
        //std::cerr << "Network error: " << e.what() << std::endl;
        global_interface.exit = 1;
        sprintf(global_interface.exit_message, "Network error: %s", e.what());
    }
}


// Load the charset from the remote host.
void CHNetworkClient::loadCharsetWithData(CHCharset *Charset) {
    char remoteCharset[MAX_CHARSET_LENGTH * MAX_PASSWORD_LEN];

    try {
        boost::system::error_code error = boost::asio::error::host_not_found;
        // Send the command to the server.
        boost::asio::write(*this->socket, boost::asio::buffer(charsetCommand, strlen(charsetCommand)),
                boost::asio::transfer_all(), error);
        // Read the data back into the remoteCharset buffer
        boost::asio::read(*this->socket, boost::asio::buffer(remoteCharset));
        
        // Load the charset into the provided charset class.
        Charset->loadRemoteCharsetIntoCharset(remoteCharset);
        
        if (error) {
            throw boost::system::system_error(error); // Some other error.
        }
    }
    catch (std::exception& e)
    {
        //std::cerr << "Network error: " << e.what() << std::endl;
        global_interface.exit = 1;
        sprintf(global_interface.exit_message, "Network error: %s", e.what());
    }
}


void CHNetworkClient::loadHashlistWithData(CHHashFileTypes *Hashlist) {
    unsigned char *remoteHashlist;


    try {
        // Allocate a new hashlist space.
        remoteHashlist = new unsigned char[this->GeneralInfo.structure.number_hashes * Hashlist->GetHashLength()];

        boost::system::error_code error = boost::asio::error::host_not_found;
        // Send the command to the server.
        boost::asio::write(*this->socket, boost::asio::buffer(hashListCommand, strlen(hashListCommand)),
                boost::asio::transfer_all(), error);
        // Read the data back into the remoteCharset buffer
        boost::asio::read(*this->socket, boost::asio::buffer(remoteHashlist,
                this->GeneralInfo.structure.number_hashes * Hashlist->GetHashLength()));

        // Load the charset into the provided charset class.
        Hashlist->importHashListFromRemoteSystem(remoteHashlist, this->GeneralInfo.structure.number_hashes);

        if (error) {
            throw boost::system::system_error(error); // Some other error.
        }
    }
    catch (std::exception& e)
    {
        //std::cerr << "Network error: " << e.what() << std::endl;
        global_interface.exit = 1;
        sprintf(global_interface.exit_message, "Network error: %s", e.what());
    }
}


void CHNetworkClient::provideGeneralInfo(CHMultiforcerNetworkGeneral *generalInfo) {
    memcpy(generalInfo, &this->GeneralInfo, sizeof(CHMultiforcerNetworkGeneral));
}



// Get a workunit from the remote host.
struct CHWorkunitRobustElement CHNetworkClient::getNextNetworkWorkunit() {
    struct CHWorkunitRobustElement localWorkunit;
    CHMultiforcerNeworkWorkunitRobust networkWorkunitRead;

    try {
        // Allocate a new workunit to return
        //localWorkunit = new CHWorkunitElement;

        boost::system::error_code error = boost::asio::error::host_not_found;

        boost::asio::write(*this->socket, boost::asio::buffer(workunitCommand, strlen(workunitCommand)),
            boost::asio::transfer_all(), error);

        boost::asio::read(*this->socket, boost::asio::buffer(networkWorkunitRead.readbuffer,
            sizeof(networkWorkunitRead.readbuffer)));

        // Copy the received workunit to the local one to return.
        memcpy(&localWorkunit, &networkWorkunitRead.WorkunitStruct, sizeof(struct CHWorkunitRobustElement));

        //printf("Workunit ID: %d\n", localWorkunit->WorkUnitID);
        //printf("Startpoint: %lu\n", localWorkunit->StartPoint);
        //printf("Endpoint: %lu\n", localWorkunit->EndPoint);

        if (error) {
            throw boost::system::system_error(error); // Some other error.
        }
    }
    catch (std::exception& e)
    {
        //std::cerr << "Network error: " << e.what() << std::endl;
        global_interface.exit = 1;
        sprintf(global_interface.exit_message, "Network error: %s", e.what());
    }

    // If the workunit comes back zeroed out, it means there are no more workunits.  Return null.
    if ((localWorkunit.WorkUnitID == 0) && (localWorkunit.StartPoint == 0) && localWorkunit.EndPoint == 0) {
        //delete localWorkunit;
        //localWorkunit = NULL;
        memset(&localWorkunit, 0, sizeof(localWorkunit));
    }
    
    return localWorkunit;
}


int CHNetworkClient::submitNetworkWorkunit(struct CHWorkunitRobustElement Workunit, uint32_t FoundPasswords) {
    CHMultiforcerNeworkWorkunitRobust networkWorkunitWrite;
    char syncBuffer;

    // Buffer for the write command + the workunit buffer
    unsigned char writeBuffer[sizeof(networkWorkunitWrite.readbuffer) + 1];

    // Add the current rate for the workunit submit
    Workunit.SearchRate = this->Display->getCurrentCrackRate();


    // Copy the submit command in.
    memcpy(writeBuffer, submitWorkunitCommand, 1);
    // Copy the entire buffer at offset 1.
    memcpy(&writeBuffer[1], &Workunit, sizeof(CHWorkunitRobustElement));


    try {
        // Copy the data into the network submission.
        boost::system::error_code error = boost::asio::error::host_not_found;

        // Inform the remote server we are sending a workunit.
        boost::asio::write(*this->socket, boost::asio::buffer(writeBuffer, sizeof(networkWorkunitWrite.readbuffer) + 1),
            boost::asio::transfer_all(), error);

        // Wait for a write back of a newline to acknowledge this.
        boost::asio::read(*this->socket, boost::asio::buffer((void *)&syncBuffer,
            sizeof(syncBuffer)));

        if (error) {
            throw boost::system::system_error(error); // Some other error.
        }
    }
    catch (std::exception& e)
    {
        //std::cerr << "Network error: " << e.what() << std::endl;
        global_interface.exit = 1;
        sprintf(global_interface.exit_message, "Network error: %s", e.what());
    }
    return 1;
}

int CHNetworkClient::reportNetworkFoundPassword(unsigned char *Hash, unsigned char *Password) {

    char syncBuffer;
    CHMultiforcerNetworkUnsaltedHashTransfer HashToSubmit;

    // Buffer for the write command + the workunit buffer
    unsigned char writeBuffer[sizeof(HashToSubmit.readbuffer) + 1];

    memcpy(HashToSubmit.Hash.hash, Hash, 32);
    memcpy(HashToSubmit.Hash.password, Password, MAX_PASSWORD_LEN);

    // Copy the submit command in.
    memcpy(writeBuffer, submitPasswordCommand, 1);
    // Copy the entire buffer at offset 1.
    memcpy(&writeBuffer[1], &HashToSubmit, sizeof(HashToSubmit));


    try {
        // Copy the data into the network submission.

        boost::system::error_code error = boost::asio::error::host_not_found;

        // Inform the remote server we are sending a workunit.
        boost::asio::write(*this->socket, boost::asio::buffer(writeBuffer, sizeof(HashToSubmit.readbuffer) + 1),
            boost::asio::transfer_all(), error);

        // Wait for a write back of a newline to acknowledge this.
        boost::asio::read(*this->socket, boost::asio::buffer((void *)&syncBuffer,
            sizeof(syncBuffer)));

        if (error) {
            throw boost::system::system_error(error); // Some other error.
        }
    }
    catch (std::exception& e)
    {
        //std::cerr << "Network error: " << e.what() << std::endl;
        global_interface.exit = 1;
        sprintf(global_interface.exit_message, "Network error: %s", e.what());
    }
    return 1;

}

void CHNetworkClient::setDisplay(CHDisplay *newDisplay) {
    this->Display = newDisplay;
}