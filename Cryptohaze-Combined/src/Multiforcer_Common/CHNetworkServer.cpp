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

#include "Multiforcer_Common/CHNetworkServer.h"
#include "Multiforcer_Common/CHNetworkCommon.h"


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



CHNetworkServer::CHNetworkServer(int port) {
    // Set the network port as requested.
    this->portNumber = port;
    this->passwordLength = 0;

    // Clear out all the pointers in the class container.
    this->classContainer.Charset = NULL;
    this->classContainer.CommandLineData = NULL;
    this->classContainer.HashFile = NULL;
    this->classContainer.Workunit = NULL;
    this->classContainer.NetworkServer = this;
    this->hashTypeId = 0;
    this->NetworkServer = NULL;
    
}

// Set the various subclasses that are needed.
// You should NOT need to override these!
void CHNetworkServer::setCommandLineData(CHCommandLineData *NewCommandLineData) {
    this->classContainer.CommandLineData = NewCommandLineData;
}
void CHNetworkServer::setCharset(CHCharset *NewCharset) {
    this->classContainer.Charset = NewCharset;
}
void CHNetworkServer::setWorkunit(CHWorkunitBase *NewWorkunit) {
    this->classContainer.Workunit = NewWorkunit;
}
void CHNetworkServer::setHashFile(CHHashFileTypes *NewHashFile) {
    this->classContainer.HashFile = NewHashFile;
}
void CHNetworkServer::setDisplay(CHDisplay *NewDisplay) {
    this->classContainer.Display = NewDisplay;
}
void CHNetworkServer::setPasswordLength(int newPasswordLength) {
    this->passwordLength = newPasswordLength;
}
int  CHNetworkServer::getPasswordLength() {
    return this->passwordLength;
}
void CHNetworkServer::setHashTypeId(int newHashTypeId) {
    this->hashTypeId = newHashTypeId;
}
int  CHNetworkServer::getHashTypeId() {
    return this->hashTypeId;
}


// Start up all the IO threads to handle traffic.
int CHNetworkServer::startNetwork() {
    int i;

    // Only create the network server if it's not already present.
    if (!this->NetworkServer) {
        this->NetworkServer = new CHNetworkServerInstance(this->io_service, this->portNumber, this->classContainer);
    }
    
    // Launch all the network IO threads
    for (i = 0; i < MAX_IO_THREADS; i++) {
        this->ioThreads[i] = new boost::thread(RunIoService, &this->io_service);
    }

    return 1;
}

// Stop the io_service instances, and wait for all the threads to return.
int CHNetworkServer::stopNetwork() {

    int i;

    this->io_service.stop();
    for (i = 0; i < MAX_IO_THREADS; i++) {
        this->ioThreads[i]->join();
    }

    return 1;
}




// CHNetworkServerSession functions
void CHNetworkServerSession::start() {
    if (this->classContainer.CommandLineData->GetDevDebug()) {
        printf("In CHNetworkServerSession::start()\n");
    }
    char remoteHostString[1024];

    this->displayThreadId = this->classContainer.Display->getFreeThreadId();
    sprintf(this->hostIpAddress, "%s", socket_.remote_endpoint().address().to_string().c_str());
    sprintf(remoteHostString, "%d: %s",  this->displayThreadId, this->hostIpAddress);
    this->classContainer.Display->addStatusLine(remoteHostString);
    this->classContainer.Display->alterNetworkClientCount(1);
    this->ClientId = this->classContainer.Workunit->GetClientId();

    // Get the thread ID for displaying speed
    
    this->classContainer.Display->setThreadCrackSpeed(this->displayThreadId, 3, 0.00);

    socket_.async_read_some(boost::asio::buffer(data_, max_length),
            boost::bind(&CHNetworkServerSession::handle_read, this,
            boost::asio::placeholders::error,
            boost::asio::placeholders::bytes_transferred));
}



void CHNetworkServerSession::handle_read(const boost::system::error_code& error,
            size_t bytes_transferred) {
    
        if (this->classContainer.CommandLineData->GetDevDebug()) {
            printf("In session::handle_read()\n");
            printf("Buffer (%d): %c\n", bytes_transferred, data_[0]);
        }
        if (!error) {
            if (bytes_transferred > 0) {
                // Do what the command requests.
                // 'h': 
                if (data_[0] == 'h') {
                    // == Hash list ==
                    if (this->classContainer.CommandLineData->GetDevDebug()) {
                        printf("Received command 'h'\n");
                        printf("Sending hash list\n");
                    }
                    this->hashList = this->classContainer.HashFile->ExportUncrackedHashList();
                    this->currentAction = NETWORK_ACTION_HASHLIST;
                    boost::asio::async_write(socket_,
                        boost::asio::buffer(this->hashList, 
                            this->classContainer.HashFile->GetUncrackedHashCount() *
                            this->classContainer.HashFile->GetHashLength()),
                        boost::bind(&CHNetworkServerSession::handle_write, this,
                        boost::asio::placeholders::error));
                } else if (data_[0] == 'c') {
                    // == Charset ==
                    if (this->classContainer.CommandLineData->GetDevDebug()) {
                        printf("Received command 'c'\n");
                        printf("Sending charset\n");
                    }
                    this->charset = this->classContainer.Charset->getCharset();
                    this->currentAction = NETWORK_ACTION_CHARSET;
                    boost::asio::async_write(socket_,
                        boost::asio::buffer(this->charset,
                            MAX_PASSWORD_LEN * MAX_CHARSET_LENGTH),
                        boost::bind(&CHNetworkServerSession::handle_write, this,
                        boost::asio::placeholders::error));
                } else if (data_[0] == 'g') {
                    // == General Data ==
                    if (this->classContainer.CommandLineData->GetDevDebug()) {
                        printf("Received command g\n");
                        printf("Sending general data.\n");
                    }
                    this->currentAction = NETWORK_ACTION_GENERAL;
                    // Set attributes
                    this->GeneralData.structure.version = NETWORK_SERVER_VERSION;
                    this->GeneralData.structure.hash_id =
                        this->classContainer.NetworkServer->getHashTypeId();
                    this->GeneralData.structure.password_length =
                        this->classContainer.NetworkServer->getPasswordLength();
                    this->GeneralData.structure.workunit_size_bits = this->classContainer.Workunit->GetWorkunitBits();
                    this->GeneralData.structure.number_hashes = this->classContainer.HashFile->GetUncrackedHashCount();
                    this->GeneralData.structure.charset_max_characters = MAX_CHARSET_LENGTH;
                    this->GeneralData.structure.charset_max_length = MAX_PASSWORD_LEN;
                    //printf("Writing %d bytes\n", sizeof(this->GeneralData.readbuffer));
                    boost::asio::async_write(socket_,
                        boost::asio::buffer(this->GeneralData.readbuffer, sizeof(this->GeneralData.readbuffer)),
                        boost::bind(&CHNetworkServerSession::handle_write, this,
                        boost::asio::placeholders::error));
                } else if (data_[0] == 'w') {
                    // == Workunit ==
                    if (this->classContainer.CommandLineData->GetDevDebug()) {
                        printf("Received command w\n");
                    }
                    this->currentAction = NETWORK_ACTION_WORKUNIT;
                    // Some ugly hacks here for now.
                    // If the number of remaining workunits is less than the number of devices,
                    // don't pass a workunit
                    // to the network client.  This allows it to "reset" and get the next
                    // password length.  This is ugly, and should be fixed. :)
                    if (this->classContainer.Workunit->GetNumberOfCompletedWorkunits() >
                            (this->classContainer.Workunit->GetNumberOfWorkunits()/* -
                            (this->classContainer.CommandLineData->GetCUDANumberDevices() + 2)*/)) {
                        if (this->classContainer.CommandLineData->GetDevDebug()) {
                            printf("Clearing workunit to return NULL!\n");
                        }
                        memset(&this->Workunit, 0, sizeof(this->Workunit));
                    } else {
                        if (this->classContainer.CommandLineData->GetDevDebug()) {
                            printf("Getting valid workunit to return.\n");
                        }
                        this->Workunit = this->classContainer.Workunit->GetNextWorkunit(this->ClientId);
                    }
                    // If a delay is requested, do it.
                    //while (this->Workunit.IsValid == WORKUNIT_PLZ_HOLD) {
                    //    CHSleep(1);
                    //    this->Workunit = this->classContainer.Workunit->GetNextWorkunit(this->ClientId);
                    //}
                    //printf("Got next workunit.\n");
                    if (!this->Workunit.IsValid) {
                        if (this->classContainer.CommandLineData->GetDevDebug()) {
                            printf("Workunit is zero!\n");
                        }
                        // Null workunit to signal end.
                        memset(&this->WorkunitData, 0, sizeof(this->WorkunitData));
                    } else {
                        // Copy values over
                        memcpy(&this->WorkunitData, &this->Workunit, sizeof(this->Workunit));
                        //this->WorkunitData.WorkunitStruct.WorkUnitID = this->Workunit->WorkUnitID;
                        //this->WorkunitData.WorkunitStruct.StartPoint = this->Workunit->StartPoint;
                        //this->WorkunitData.WorkunitStruct.EndPoint = this->Workunit->EndPoint;
                    }
                    //printf("Set workunitData to workunit\n");
                    boost::asio::async_write(socket_,
                        boost::asio::buffer(this->WorkunitData.readbuffer, sizeof(this->WorkunitData.readbuffer)),
                        boost::bind(&CHNetworkServerSession::handle_write, this,
                        boost::asio::placeholders::error));
                } else if (data_[0] == 's') {
                    // == Submit workunit ==
                    CHMultiforcerNeworkWorkunitRobust WorkunitInfo;
                    if (this->classContainer.CommandLineData->GetDevDebug()) {
                        printf("Received command s\n");
                        printf("Want to read %d bytes\n", sizeof(WorkunitInfo.readbuffer));
                    }
                    // Copy memory into workunit buffer
                    if (bytes_transferred >= sizeof(WorkunitInfo.readbuffer)) {
                        memcpy(WorkunitInfo.readbuffer, &data_[1], sizeof(WorkunitInfo.readbuffer));

                        //boost::asio::read(socket_, boost::asio::buffer(WorkunitInfo.readbuffer, sizeof(WorkunitInfo.readbuffer)));
                        // Should have the submitted workunit now.
                        //printf("Received workunit ID: %d\n", WorkunitInfo.WorkunitStruct.WorkUnitID);
                        //printf("Startpoint: %lu\n", WorkunitInfo.WorkunitStruct.StartPoint);
                        //printf("Endpoint: %lu\n", WorkunitInfo.WorkunitStruct.EndPoint);

                        //printf("Passwords found: %d\n", WorkunitInfo.WorkunitStruct.PasswordsFound);
                        //printf("Time to complete: %f\n", WorkunitInfo.WorkunitStruct.SecondsToFinish);
                        // Now "submit" the workunit
                        CHWorkunitRobustElement WU;
                        //WU = new CHWorkunitElement;
                        // Copy the data
                        WU.EndPoint = WorkunitInfo.WorkunitStruct.EndPoint;
                        WU.PasswordsFound = WorkunitInfo.WorkunitStruct.PasswordsFound;
                        //WU->SecondsToFinish = WorkunitInfo.WorkunitStruct.SecondsToFinish;
                        WU.StartPoint = WorkunitInfo.WorkunitStruct.StartPoint;
                        WU.WorkUnitID = WorkunitInfo.WorkunitStruct.WorkUnitID;
                        // Submit the workunit.  This deletes the struct.
                        this->classContainer.Workunit->SubmitWorkunit(WU);
                        this->classContainer.Display->setWorkunitsCompleted(this->classContainer.Workunit->GetNumberOfCompletedWorkunits());
                        this->classContainer.Display->setThreadCrackSpeed(this->displayThreadId, 3, WorkunitInfo.WorkunitStruct.SearchRate);
                        //printf("Done submitting workunit.\n");
                        char newlineSync;
                        newlineSync = ' ';
                        boost::asio::async_write(socket_,
                            boost::asio::buffer((void *)&newlineSync, sizeof(newlineSync)),
                            boost::bind(&CHNetworkServerSession::handle_write, this,
                            boost::asio::placeholders::error));
                    }
                } else if (data_[0] == 'p') {
                    if (this->classContainer.CommandLineData->GetDevDebug()) {
                        printf("Received command p\n");
                        printf("Want to read %d bytes\n", sizeof(CHMultiforcerNetworkUnsaltedHashTransfer));
                    }
                    if (bytes_transferred >= sizeof(CHMultiforcerNetworkUnsaltedHashTransfer)) {

                        CHMultiforcerNetworkUnsaltedHashTransfer NewFoundHash;
                        int i;
                        unsigned char *Password, *Hash;
                        
                        memcpy(NewFoundHash.readbuffer, &data_[1], sizeof(NewFoundHash.readbuffer));
                        if (this->classContainer.CommandLineData->GetDevDebug()) {
                            printf("Got hash: ");
                            for (i = 0; i < 16; i++) {
                                printf("%02x", NewFoundHash.Hash.hash[i]);
                            }
                            printf("\n");
                            printf("Password: %s\n", NewFoundHash.Hash.password);
                        }

                        Password = new unsigned char[MAX_PASSWORD_LEN];
                        Hash = new unsigned char[32];

                        memcpy(Password, NewFoundHash.Hash.password, MAX_PASSWORD_LEN);
                        memcpy(Hash, NewFoundHash.Hash.hash, 32);

                        this->classContainer.HashFile->ReportFoundPassword(Hash, Password);
                        this->classContainer.Display->addCrackedPassword((char *)Password);
                        this->classContainer.Display->setCrackedHashes(this->classContainer.HashFile->GetCrackedHashCount());
                        this->classContainer.Display->Refresh();

                        delete[] Password;
                        delete[] Hash;
                        

                        char newlineSync;
                        newlineSync = ' ';
                        boost::asio::async_write(socket_,
                            boost::asio::buffer((void *)&newlineSync, sizeof(newlineSync)),
                            boost::bind(&CHNetworkServerSession::handle_write, this,
                            boost::asio::placeholders::error));
                    }

                } else{
                    if (this->classContainer.CommandLineData->GetDevDebug()) {
                        printf("Unknown command\n");
                    }
                }
            }

        } else {
            // Report the disconnect
            char remoteHostString[1024];
            sprintf(remoteHostString, "DSC: %s", this->hostIpAddress);
            this->classContainer.Display->addStatusLine(remoteHostString);
            this->classContainer.Display->alterNetworkClientCount(-1);
            this->classContainer.Display->setThreadCrackSpeed(this->displayThreadId, 0, 0.00);
            this->classContainer.Workunit->CancelAllWorkunitsByClientId(this->ClientId);
            this->classContainer.Display->setWorkunitsCompleted(this->classContainer.Workunit->GetNumberOfCompletedWorkunits());
            this->classContainer.Workunit->FreeClientId(this->ClientId);


            delete this;
        }
}

void CHNetworkServerSession::handle_write(const boost::system::error_code& error) {
    if (this->classContainer.CommandLineData->GetDevDebug()) {
        printf("In session::handle_write()\n");
        printf("Current action: %d\n", this->currentAction);
    }

    if (this->currentAction == NETWORK_ACTION_HASHLIST) {
        // If the action was a hash list, delete the send buffer.
        delete[] this->hashList;
    } else if (this->currentAction == NETWORK_ACTION_CHARSET) {
        delete[] this->charset;
    }
    this->currentAction = 0;


    if (!error) {
        socket_.async_read_some(boost::asio::buffer(data_, max_length),
                boost::bind(&CHNetworkServerSession::handle_read, this,
                boost::asio::placeholders::error,
                boost::asio::placeholders::bytes_transferred));
    } else {
        delete this;
    }
}

