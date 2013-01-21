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


// Main code for the Cryptohaze Multiforcer.



#include "Multiforcer_Common/CHHashes.h"
#include "Multiforcer_CUDA_host/CHCommandLineData.h"
#include "CUDA_Common/CHCudaUtils.h"
#include "CH_Common/CHWorkunitRobust.h"
#include "CH_Common/CHWorkunitNetwork.h"
#include "CH_Common/CHCharsetSingle.h"
#include "CH_Common/CHCharsetMulti.h"
#include "CH_Common/CHHashFilePlain32.h"
#include "CH_Common/CHHashFileSalted32.h"
#include "CH_Common/CHHashFileMSSQL.h"
#include "CH_Common/CHHashFilePlainSHA.h"
#include "CH_Common/CHHashFileSaltedSSHA.h"

#include "Multiforcer_Common/CHDisplayNcurses.h"
#include "Multiforcer_Common/CHDisplayDebug.h"
#include "Multiforcer_Common/CHDisplayDaemon.h"
#include "Multiforcer_CUDA_host/CHHashTypePlainMD5.h"
#include "Multiforcer_CUDA_host/CHHashTypePlainMD4.h"
#include "Multiforcer_CUDA_host/CHHashTypePlainNTLM.h"
#include "Multiforcer_CUDA_host/CHHashTypePlainSHA1.h"
#include "Multiforcer_CUDA_host/CHHashTypeMSSQL.h"
#include "Multiforcer_CUDA_host/CHHashTypePlainMySQL323.h"
#include "Multiforcer_CUDA_host/CHHashTypeSaltedMD5PassSalt.h"
#include "Multiforcer_CUDA_host/CHHashTypeSaltedMD5SaltPass.h"
#include "Multiforcer_CUDA_host/CHHashTypeSaltedSSHA.h"
#include "Multiforcer_CUDA_host/CHHashTypePlainMD5Single.h"

#include "Multiforcer_CUDA_host/CHHashTypePlainMD5ofSHA1.h"

#if USE_NETWORK
#include "Multiforcer_Common/CHNetworkServer.h"
#include "Multiforcer_Common/CHNetworkClient.h"
#endif

// global_commands is a way of communicating across all threads.
// It handles exit requests and error handling.
struct global_commands global_interface;

// Ctrl-C handler.  Terminate cleanly.
void terminate_process(int sig) {
    // Set the exit value to 1 to force all threads to exit.
    global_interface.exit = 1;
    global_interface.user_exit = 1;
}

// Runs the multiforcer in standalone or network server mode.
void runStandaloneOrServerMode(CHCommandLineData *CommandLineData) {
    int i;

    CHCharset *Charset;
    CHWorkunitBase *RobustWorkunit;
    CHHashFileTypes *HashFile;
    CHDisplay *Display;
    char printBuffer[1000];
    CHHashes HashTypes;

    CHHashType *HashType;
    uint64_t NumberOfPasswords;

    int maxPasswordLength = 0;

    // Default size.  May be overridden.
    int WorkunitSize;

    std::vector<uint8_t> RestoreData;

    std::string ResumeFilename;

    {
        char ResumeTimestampBuffer[1024];
        struct timeval resume_time;
        time_t resume_time_t;
        // Get the resume filename with timestamp.
        gettimeofday(&resume_time, NULL);
        resume_time_t=resume_time.tv_sec;
        memset(ResumeTimestampBuffer, 0, sizeof(ResumeTimestampBuffer));
        strftime(ResumeTimestampBuffer, 128, "%Y-%m-%d-%H-%M-%S", localtime(&resume_time_t));
        ResumeFilename = "CM-Resume-";
        ResumeFilename += ResumeTimestampBuffer;
        ResumeFilename += ".mfr";
    }


#if USE_NETWORK
    CHNetworkServer *Network;
#endif

    RobustWorkunit = new CHWorkunitRobust();

    if (CommandLineData->GetDevDebug()) {
        RobustWorkunit->EnableDebugOutput();
    }

    if (CommandLineData->GetUseRestoreFile()) {
        if (!RobustWorkunit->LoadStateFromFile(CommandLineData->GetRestoreFileName())) {
            printf("Loading state from file failed.\n");
            exit(1);
        }
        RestoreData = RobustWorkunit->GetResumeMetadata();
        CommandLineData->SetDataFromRestore(RestoreData);
        // Overwrite the existing one as we progress.
        RobustWorkunit->SetResumeFile(ResumeFilename);
    } else {
        RobustWorkunit->SetResumeFile(ResumeFilename);
    }


    // Set the hash type being used.
    HashTypes.SetHashId(CommandLineData->GetHashType());

    // Get the HashType class and HashFile class

    HashType = HashTypes.GetHashType();
    HashFile = HashTypes.GetHashFile();
    WorkunitSize = HashTypes.GetDefaultWorkunitSizeBits();

    // If either comes back null, something has gone very wrong.
    if (!HashType || !HashFile) {
        printf("Invalid hash type selected!\n");
        exit(1);
    }

    // If the workunit size was set on the command line, use it here.
    if (CommandLineData->GetWorkunitBits()) {
        WorkunitSize = CommandLineData->GetWorkunitBits();
    }

    // If an output file is to be used, set it here.
    if (CommandLineData->GetUseOutputFile()) {
        HashFile->SetFoundHashesOutputFilename(CommandLineData->GetOutputFileName());
    }

    if (CommandLineData->GetUseCharsetMulti()) {
        Charset = new CHCharsetMulti();
        if (!Charset->getCharsetFromFile(CommandLineData->GetCharsetFileName())) {
            printf("Cannot open charset!\n");
            exit(1);
        }
    } else {
        CHCharset *CharsetSingle;
        CharsetSingle = new CHCharsetSingle();
        if (!CharsetSingle->getCharsetFromFile(CommandLineData->GetCharsetFileName())) {
            printf("Cannot open charset!\n");
            exit(1);
        }
        Charset = new CHCharsetMulti();
        Charset->convertSingleCharsetToMulti(CharsetSingle->getCharset(), CharsetSingle->getCharsetLength(0));
    }

    if (!HashFile->OpenHashFile(CommandLineData->GetHashListFileName())) {
        printf("Cannot open hash file!\n");
        exit(1);
    }
    
    // Add hex output option if desired.
    HashFile->SetAddHexOutput(CommandLineData->GetAddHexOutput());


    HashType->setCharset(Charset);
    HashType->setCommandLineData(CommandLineData);
    HashType->setHashFile(HashFile);
    HashType->setWorkunit(RobustWorkunit);

    // If we are using debug, set the display to that mode.
    if (CommandLineData->GetDebug()) {
        Display = new CHMultiforcerDisplayDebug();
    } else {
        // Normal curses output
        Display = new CHMultiforcerDisplay();
    }

#if USE_NETWORK
    Network = NULL;
    // If requested bring the network online and assign types to it
    if (CommandLineData->GetIsNetworkServer()) {
        Network = new CHNetworkServer(CommandLineData->GetNetworkPort());

        Network->setCharset(Charset);
        Network->setCommandLineData(CommandLineData);
        Network->setHashFile(HashFile);
        Network->setWorkunit(RobustWorkunit);
        Network->setDisplay(Display);
        Network->setHashTypeId(CommandLineData->GetHashType());

        Network->startNetwork();

        // Update the display with the server info.
        char portBuffer[16];
        sprintf(portBuffer, "%d", CommandLineData->GetNetworkPort());
        Display->setSystemMode(SYSTEM_MODE_SERVER, portBuffer);
    }
#endif

    HashType->setDisplay(Display);

    Display->setHashName(HashTypes.GetHashStringFromID(CommandLineData->GetHashType()));

    sprintf(printBuffer, "chr isMulti: %d", Charset->getIsMulti());
    Display->addStatusLine(printBuffer);


    // Add a few GPU threads
    if (CommandLineData->GetDebug()) {
        // If debug is in use, use the CUDA device ID.  Default 0.
        HashType->addGPUDeviceID(CommandLineData->GetCUDADevice());
    } else {
        for (i = 0; i < CommandLineData->GetCUDANumberDevices(); i++) {
            HashType->addGPUDeviceID(i);
            Display->setThreadCrackSpeed(i, GPU_THREAD, 0.00);
        }
    }

    // Catch Ctrl-C and handle it gracefully
    signal(SIGINT, terminate_process);

    // If a max length has been set, use it.
    // Otherwise just set to the max supported length.
    if (CommandLineData->GetMaxPasswordLength()) {
        maxPasswordLength = CommandLineData->GetMaxPasswordLength();
    } else {
        maxPasswordLength = HashTypes.GetMaxSupportedLength();
    }

    for (i = CommandLineData->GetMinPasswordLength(); i <= maxPasswordLength; i++) {
        // Set the status line to indicate where we are.
        sprintf(printBuffer, "Starting pw len %d", i);
        Display->addStatusLine(printBuffer);

        // If no hashes are left, exit.
        if (HashFile->GetUncrackedHashCount() == 0) {
            global_interface.exit = 1;
            strcpy(global_interface.exit_message, "All hashes found!  Exiting!\n");
            break;
        }

#if USE_NETWORK
        // Set the network support for the password length
        if (CommandLineData->GetIsNetworkServer()) {
            Network->setPasswordLength(i);
        }
#endif
        NumberOfPasswords = Charset->getPasswordSpaceSize(i);
        if (global_interface.exit) {
            break;
        }

        // Provide the correct metadata to the workunit class
        RestoreData = CommandLineData->GetRestoreData(i);
        RobustWorkunit->SetResumeMetadata(RestoreData);

        // If we are NOT restoring, create new workunits.
        if (!CommandLineData->GetUseRestoreFile()) {
            RobustWorkunit->CreateWorkunits(NumberOfPasswords, WorkunitSize, i);
        }

        if (global_interface.exit) {
            break;
        }

        Display->setWorkunitsTotal(RobustWorkunit->GetNumberOfWorkunits());
        Display->setWorkunitsCompleted(RobustWorkunit->GetNumberOfCompletedWorkunits());

        HashType->crackPasswordLength(i);

        if (global_interface.exit) {
            break;
        }
    }

    delete RobustWorkunit;
    delete Display;
    
    HashFile->PrintAllFoundHashes();
    // If we are outputting unfound hashes, do it now.
    if (CommandLineData->GetUseUnfoundOutputFile()) {
        HashFile->OutputUnfoundHashesToFile(CommandLineData->GetUnfoundOutputFileName());
    }

}


// Runs a network client mode.
void runNetworkClientMode(CHCommandLineData *CommandLineData) {
    // We only want this to compile if network support is enabled.
#if USE_NETWORK
    CHNetworkClient *NetworkClient;
    CHCharset *Charset;
    CHHashType *HashType;
    CHWorkunitNetworkClient *WorkunitNetworkClient;
    CHHashFileTypes *HashFile;
    CHDisplay *Display;
    CHHashes HashTypes;

    char printBuffer[1000];
    int i;

    // If we are a network client, get data that way.
    NetworkClient = NULL;
    WorkunitNetworkClient = new CHWorkunitNetworkClient;
    CHMultiforcerNetworkGeneral NetworkGeneralData;

    while (1) {
        // Reset exit flag - new attempt.
        global_interface.exit = 0;
        NetworkClient = new CHNetworkClient(CommandLineData->GetNetworkRemoteHostname(), CommandLineData->GetNetworkPort());
        Charset = new CHCharsetMulti();
        NetworkClient->loadCharsetWithData(Charset);

        NetworkClient->provideGeneralInfo(&NetworkGeneralData);

        HashTypes.SetHashId(NetworkGeneralData.structure.hash_id);


        // Check to see if the hash type is supported as a network client.
        if (!HashTypes.GetIsNetworkSupported()) {
            printf("Hash type %d not supported in network client!\n", NetworkGeneralData.structure.hash_id);
            continue;
        }

        HashType = HashTypes.GetHashType();
        HashFile = HashTypes.GetHashFile();

        // If either comes back null, something has gone very wrong.
        if (!HashType || !HashFile) {
            printf("Invalid hash type selected!\n");
            continue;
        }

        HashFile->setNetworkClient(NetworkClient);

        // Load the main hash list
        NetworkClient->loadHashlistWithData(HashFile);

        // Load the workunit with the network client.
        WorkunitNetworkClient->setNetworkClient(NetworkClient);

        HashType->setCharset(Charset);
        HashType->setCommandLineData(CommandLineData);
        HashType->setHashFile(HashFile);
        HashType->setWorkunit(WorkunitNetworkClient);

        if (CommandLineData->GetDebug()) {
            Display = new CHMultiforcerDisplayDebug();
        } else if (CommandLineData->GetDaemon()) {
            Display = new CHMultiforcerDisplayDaemon();
        } else {
            Display = new CHMultiforcerDisplay();
        }

        Display->setHashName(HashTypes.GetHashStringFromID(NetworkGeneralData.structure.hash_id));


        NetworkClient->setDisplay(Display);
        HashType->setDisplay(Display);
        Display->setSystemMode(SYSTEM_MODE_CLIENT, CommandLineData->GetNetworkRemoteHostname());

        // Add a few GPU threads
        if (CommandLineData->GetDebug()) {
            HashType->addGPUDeviceID(CommandLineData->GetCUDADevice());
        } else {
            for (i = 0; i < CommandLineData->GetCUDANumberDevices(); i++) {
                HashType->addGPUDeviceID(i);
            }
        }


        while(!global_interface.exit) {
            sprintf(printBuffer, "Waiting on server.");
            Display->addStatusLine(printBuffer);
            CHSleep(NETWORK_WAIT_TIME);
            NetworkClient->updateGeneralInfo();
            NetworkClient->provideGeneralInfo(&NetworkGeneralData);
            sprintf(printBuffer, "Pw len: %d", NetworkGeneralData.structure.password_length);
            Display->addStatusLine(printBuffer);
            WorkunitNetworkClient->setPasswordLength(NetworkGeneralData.structure.password_length);
            HashType->crackPasswordLength(NetworkGeneralData.structure.password_length);
        }

        delete Display;
        delete NetworkClient;
        delete Charset;
        delete HashType;

        HashFile->PrintAllFoundHashes();

        delete HashFile;
        
        // If the user wants to quit, quit.
        if (global_interface.user_exit) {
            return;
        }   
    }
#endif
}

int main(int argc, char *argv[]) {

    CHCommandLineData CommandLineData;

    // Init the global stuff
    global_interface.exit = 0;
    global_interface.user_exit = 0;
    global_interface.pause = 0;
    memset(global_interface.exit_message, 0, sizeof(global_interface.exit_message));


    // Get the command line data.  If not success, fail.
    if (!CommandLineData.ParseCommandLine(argc, argv)) {
        exit(1);
    }

    // Catch Ctrl-C and handle it gracefully
    signal(SIGINT, terminate_process);

    // Run the appropriate mode
    if (CommandLineData.GetIsNetworkClient()) {
        runNetworkClientMode(&CommandLineData);
    } else {
        runStandaloneOrServerMode(&CommandLineData);
    }

    // If there is a message to print, terminate.
    if (strlen(global_interface.exit_message)) {
        printf("\n\nTerminating due to error: %s\n", global_interface.exit_message);
    }
}
