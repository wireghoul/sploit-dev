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

#include "MFN_Common/MFNRun.h"
#include "MFN_Common/MFNCommandLineData.h"


// Ctrl-C handler.  Terminate cleanly.
void terminate_process(int sig) {
    // Set the exit value to 1 to force all threads to exit.
    global_interface.exit = 1;
    global_interface.user_exit = 1;
}

int main(int argc, char *argv[]) {

    MFNCommandLineData *CommandLineData;

    // Init the global stuff
    global_interface.exit = 0;
    global_interface.user_exit = 0;
    global_interface.pause = 0;
    memset(global_interface.exit_message, 0, sizeof(global_interface.exit_message));

    CommandLineData = MultiforcerGlobalClassFactory.getCommandlinedataClass();
    MultiforcerGlobalClassFactory.setDisplayClassType(MFN_DISPLAY_CLASS_CURSES);

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
