// New Multiforcer main file.



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



#include "MFN_Common/MFNCommandLineData.h"
#include "CUDA_Common/CHCudaUtils.h"
#include "MFN_Common/MFNWorkunitRobust.h"
#include "MFN_Common/MFNWorkunitWordlist.h"
#include "MFN_Common/MFNWorkunitNetwork.h"

#include "MFN_Common/MFNDisplayDebug.h"
#include "MFN_Common/MFNDisplayCurses.h"
#include "MFN_Common/MFNDisplayDaemon.h"

#include "CH_Common/CHCharsetNew.h"
#include "MFN_Common/MFNHashType.h"
#include "MFN_Common/MFNHashTypePlain.h"
#include "MFN_CUDA_host/MFNHashTypePlainCUDA_MD5.h"
#include "MFN_OpenCL_host/MFNHashTypePlainOpenCL_MD5.h"
#include "CH_HashFiles/CHHashFileVPlain.h"

#include "MFN_Common/MFNMultiforcerClassFactory.h"
#include "MFN_Common/MFNHashClassLauncher.h"

#include "MFN_Common/MFNNetworkServer.h"
#include "MFN_Common/MFNNetworkClient.h"

#include "CUDA_Common/CHCudaUtils.h"

#include "MFN_Common/MFNGeneralInformation.h"

#include "MFN_Common/MFNRun.h"

// global_commands is a way of communicating across all threads.
// It handles exit requests and error handling.
struct global_commands global_interface;

/**
 * Global class factory.
 */
MFNClassFactory MultiforcerGlobalClassFactory;

// Needed for the Windows side of things.
void MFNRun()
{
    if (MultiforcerGlobalClassFactory.getCommandlinedataClass()->GetIsNetworkClient()) {
        runNetworkClientMode();
    } else {
        runStandaloneOrServerMode();
    }
}

// Runs the multiforcer in standalone or network server mode.
void runStandaloneOrServerMode() {
    int i;

    CHCharsetNew *Charset;
    MFNWorkunitBase *Workunit;
    CHHashFileV *HashFile;
    MFNDisplay *Display;
    char printBuffer[1000];
    MFNHashClassLauncher HashClassLauncher;
    MFNHashIdentifiers *HashIdentifiers;
    MFNNetworkServer *NetworkServer = NULL;
    MFNGeneralInformation *GeneralInformation;
    MFNCommandLineData *CommandLineData;
    
    
    uint32_t hashId;

    int maxPasswordLength = 0;
    
    CommandLineData = MultiforcerGlobalClassFactory.getCommandlinedataClass();

    // Default size.  May be overridden.
    int WorkunitSize = 32;
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

    // Determine the hash type
    HashIdentifiers = MultiforcerGlobalClassFactory.getHashIdentifiersClass();
    if (!HashIdentifiers) {
        printf("Cannot get hash identifiers class!\n");
        exit(1);
    }
    hashId = HashIdentifiers->GetHashIdFromString(CommandLineData->GetHashTypeString());
    if (hashId == MFN_HASHTYPE_UNDEFINED) {
        printf("Invalid hash type %s!\n", CommandLineData->GetHashTypeString().c_str());
        HashIdentifiers->PrintAllHashTypes();
        exit(1);
    }

    GeneralInformation = MultiforcerGlobalClassFactory.getGeneralInformationClass();
    GeneralInformation->setCharsetClassId(CH_CHARSET_NEW_CLASS_ID);
    GeneralInformation->setHashId(hashId);
    GeneralInformation->setPasswordLength(CommandLineData->GetMinPasswordLength());

    
    // Get our classes
    MultiforcerGlobalClassFactory.setCharsetClassType(CH_CHARSET_NEW_CLASS_ID);
    if (CommandLineData->GetDebug()) {
        MultiforcerGlobalClassFactory.setDisplayClassType(MFN_DISPLAY_CLASS_DEBUG);
    } else if (CommandLineData->GetDaemon()) {
        MultiforcerGlobalClassFactory.setDisplayClassType(MFN_DISPLAY_CLASS_DAEMON);
    } else {
        MultiforcerGlobalClassFactory.setDisplayClassType(MFN_DISPLAY_CLASS_CURSES);
    }
    
    // If this is a wordlist class, use a wordlist workunit.
    if (HashIdentifiers->GetHasWordlist()) {
        MultiforcerGlobalClassFactory.setWorkunitClassType(MFN_WORKUNIT_WORDLIST_CLASS_ID);
    } else {
        MultiforcerGlobalClassFactory.setWorkunitClassType(MFN_WORKUNIT_ROBUST_CLASS_ID);
    }

    // Set up the hash specific stuff.
    MultiforcerGlobalClassFactory.setHashfileClassType(HashIdentifiers->GetHashData().HashFileIdentifier);
    HashClassLauncher.setHashType(HashIdentifiers->GetHashData().HashTypeIdentifier);

    
    Charset = MultiforcerGlobalClassFactory.getCharsetClass();
    if (!Charset) {
        printf("Cannot get charset class!\n");
        exit(1);
    }

    Workunit = MultiforcerGlobalClassFactory.getWorkunitClass();
    if (!Workunit) {
        printf("Cannot get workunit class!\n");
        exit(1);
    }
    HashFile = MultiforcerGlobalClassFactory.getHashfileClass();
    if (!HashFile) {
        printf("Cannot get hashfile class!\n");
        exit(1);
    }

    Display = MultiforcerGlobalClassFactory.getDisplayClass();
    if (!Display) {
        printf("Cannot get display class!\n");
        exit(1);
    }
    

    if (CommandLineData->GetDevDebug()) {
        Workunit->EnableDebugOutput();
    }
    
    

/*
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
*/

    // Set the hash type being used.
    //HashTypes.SetHashId(CommandLineData->GetHashType());

    // Get the HashType class and HashFile class

    /*


*/

    // If an output file is to be used, set it here.
    if (CommandLineData->GetOutputFileName().length()) {
        HashFile->SetFoundHashesOutputFilename(CommandLineData->GetOutputFileName());
    }
    // If the workunit size was set on the command line, use it here.
    if (CommandLineData->GetWorkunitBits()) {
        WorkunitSize = CommandLineData->GetWorkunitBits();
    } else {
        WorkunitSize = HashIdentifiers->GetDefaultWorkunitSizeBits();
    }

    if (!Charset->readCharsetFromFile(CommandLineData->GetCharsetFileName())) {
        printf("Cannot open charset!\n");
        exit(1);
    }
    //printf("Charset opened properly.\n");
    
    if (!HashFile->OpenHashFile(CommandLineData->GetHashListFileName())) {
        printf("Cannot open hash file!\n");
        exit(1);
    }
    //printf("Hashfile opened properly.\n");
    

    // Add hex output option if desired.
    HashFile->SetAddHexOutput(CommandLineData->GetAddHexOutput());
    HashFile->setPrintAlgorithm(CommandLineData->GetPrintAlgorithms());

    // If requested bring the network online and assign types to it
    if (CommandLineData->GetIsNetworkServer()) {
        MultiforcerGlobalClassFactory.setNetworkServerPort(CommandLineData->GetNetworkPort());
        NetworkServer = MultiforcerGlobalClassFactory.getNetworkServerClass();

        NetworkServer->startNetwork();

        // Update the display with the server info.
        char portBuffer[16];
        sprintf(portBuffer, "%d", CommandLineData->GetNetworkPort());
        Display->setSystemMode(SYSTEM_MODE_SERVER, std::string(portBuffer));
    }

    Display->setHashName(HashIdentifiers->GetHashData().HashDescriptor);
    
    if (!CommandLineData->GetIsServerOnly()) {
        if (!HashClassLauncher.addAllDevices(CommandLineData->GetDevicesToUse())) {
            printf("Cannot add devices!\n");
            exit(1);
        }
    }


    // Pick the desired max length.
    if (CommandLineData->GetMaxPasswordLength()) {
        maxPasswordLength = CommandLineData->GetMaxPasswordLength();
    } else {
        maxPasswordLength = HashIdentifiers->GetMaxSupportedLength();
    }
    
    for (i = CommandLineData->GetMinPasswordLength(); i <= maxPasswordLength; i++) {
        uint64_t NumberOfPasswords;
        
        GeneralInformation->setPasswordLength(i);
        Display->setPasswordLen(i);
        // Set the status line to indicate where we are.
        sprintf(printBuffer, "Starting pw len %d", i);
        Display->addStatusLine(printBuffer);

        // If no hashes are left, exit.
        if (HashFile->GetUncrackedHashCount() == 0) {
            global_interface.exit = 1;
            strcpy(global_interface.exit_message, "All hashes found!  Exiting!\n");
            break;
        }
        
        NumberOfPasswords = Charset->getPasswordSpaceSize(i);
        // Check for errors from charset and break.
        if (global_interface.exit) {
            break;
        }
/*
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
*/
        Workunit->CreateWorkunits(NumberOfPasswords, WorkunitSize, i);
        
        // If there are threads running locally, launch them.
        if (!CommandLineData->GetIsServerOnly()) {
            HashClassLauncher.launchThreads(i);
        }
        // Wait until all the workunits are completed.  This is useful in
        // server-only mode.
        while (Workunit->GetNumberOfCompletedWorkunits() < Workunit->GetNumberOfWorkunits()) {
            CHSleep(1);
            Display->Refresh();
            // Make termination work properly for the server
            if (global_interface.exit) {
                // Break from the while loop.
                break;
            }
        }

        // Break from the password length loop.
        if (global_interface.exit) {
            break;
        }
    }

    delete Workunit;
    delete Display;

    HashFile->PrintAllFoundHashes();
    
    // If we are outputting unfound hashes, do it now.
    if (CommandLineData->GetUnfoundOutputFileName().length()) {
        HashFile->OutputUnfoundHashesToFile(CommandLineData->GetUnfoundOutputFileName());
    }
}


/**
 * This function will wait for a server to arrive, do all the setup for the
 * instance, and will continue until the server disconnects, at which point this
 * function will return.
 */
void runNetworkOneConnect() {
    MFNCommandLineData *CommandLineData;
    MFNNetworkClient *NetworkClient;
    MFNHashIdentifiers *HashIdentifiers;
    MFNGeneralInformation *GeneralInformation;
    MFNDisplay *Display;
    MFNHashClassLauncher HashClassLauncher;

    CommandLineData = MultiforcerGlobalClassFactory.getCommandlinedataClass();
    GeneralInformation = MultiforcerGlobalClassFactory.getGeneralInformationClass();
    HashIdentifiers = MultiforcerGlobalClassFactory.getHashIdentifiersClass();
    
    // Set up the network client class.  This will wait until a connection is
    // created, or until the user exits.
    MultiforcerGlobalClassFactory.setNetworkClientPort(CommandLineData->GetNetworkPort());
    MultiforcerGlobalClassFactory.setNetworkClientRemoteHost(CommandLineData->GetNetworkRemoteHostname());
    MultiforcerGlobalClassFactory.setNetworkClientOneshot(0);
    NetworkClient = MultiforcerGlobalClassFactory.getNetworkClientClass();
   
    // If the user has requested an exit, we can just terminate now.
    if (global_interface.user_exit) {
        exit(0);
    }
    
    // Update the general information structure to continue setup.
    NetworkClient->updateGeneralInfo();
    
    // Set the hash ID from the server, and continue setting up the system.
    HashIdentifiers->SetHashId(GeneralInformation->getHashId());
    
    MultiforcerGlobalClassFactory.setCharsetClassType(CH_CHARSET_NEW_CLASS_ID);
    
    if (CommandLineData->GetDebug()) {
        MultiforcerGlobalClassFactory.setDisplayClassType(MFN_DISPLAY_CLASS_DEBUG);
    } else if (CommandLineData->GetDaemon()) {
        MultiforcerGlobalClassFactory.setDisplayClassType(MFN_DISPLAY_CLASS_DAEMON);
    } else {
        MultiforcerGlobalClassFactory.setDisplayClassType(MFN_DISPLAY_CLASS_CURSES);
    }
    
    MultiforcerGlobalClassFactory.setWorkunitClassType(MFN_WORKUNIT_NETWORK_CLASS_ID);

    // Set up the hash specific stuff based on the general information.
    MultiforcerGlobalClassFactory.setHashfileClassType(HashIdentifiers->GetHashData().HashFileIdentifier);
    
    HashClassLauncher.setHashType(HashIdentifiers->GetHashData().HashTypeIdentifier);
    
    NetworkClient->updateCharset();

    // TODO: This should handle non-existent devices cleanly.
    if (!HashClassLauncher.addAllDevices(CommandLineData->GetDevicesToUse())) {
        printf("Cannot add devices!\n");
        exit(1);
    }

    // Display online - no more output!
    Display = MultiforcerGlobalClassFactory.getDisplayClass();
    Display->setHashName(HashIdentifiers->GetHashData().HashDescriptor);


    // Loop until the server has gone away
    while (!global_interface.exit) {
        // Get the password length/etc.
        NetworkClient->updateGeneralInfo();
        
        MultiforcerGlobalClassFactory.getWorkunitClass()->
                setPasswordLength(GeneralInformation->getPasswordLength());
        
        Display->setPasswordLen(GeneralInformation->getPasswordLength());

        NetworkClient->updateUncrackedHashes();
        
        // Only run if there are hashes to crack.
        if (MultiforcerGlobalClassFactory.getHashfileClass()->GetUncrackedHashCount()) {
            HashClassLauncher.launchThreads(GeneralInformation->getPasswordLength());
        } else {
            // No hashes left - why should we continue to try and run?
            CHSleep(5);
            break;
        }
    }
    // Reset for future runs.
    global_interface.exit = 0;
    
    // Out of the main loop - clean up.
    MultiforcerGlobalClassFactory.destroyCharsetClass();
    MultiforcerGlobalClassFactory.destroyHashfileClass();
    MultiforcerGlobalClassFactory.destroyNetworkClientClass();
    MultiforcerGlobalClassFactory.destroyWorkunitClass();
    MultiforcerGlobalClassFactory.destroyDisplayClass();
}


void runNetworkClientMode() {
    printf("Network Client Mode Enabled!\n");
    
    while (!global_interface.user_exit) {
        runNetworkOneConnect();
    }
    
}

/*
 * INT MAIN IS NO LONGER HERE!  It lives in MFNGuiMain or MFNDisplayCliMain!
 * 
int main(int argc, char *argv[]) {

    MFNCommandLineData *CommandLineData;

    // Init the global stuff
    global_interface.exit = 0;
    global_interface.user_exit = 0;
    global_interface.exit_after_count = 0;
    global_interface.pause = 0;
    memset(global_interface.exit_message, 0, sizeof(global_interface.exit_message));

    CommandLineData = MultiforcerGlobalClassFactory.getCommandlinedataClass();

    // Get the command line data.  If not success, fail.
    if (!CommandLineData->ParseCommandLine(argc, argv)) {
        exit(1);
    }

    // Catch Ctrl-C and handle it gracefully
    signal(SIGINT, terminate_process);
    
    if (CommandLineData->GetIsNetworkClient()) {
        runNetworkClientMode();
    } else {
        runStandaloneOrServerMode();
    }

  
    // If there is a message to print, terminate.
    if (strlen(global_interface.exit_message)) {
        printf("\n\nTerminating due to error: %s\n", global_interface.exit_message);
    }
}
*/