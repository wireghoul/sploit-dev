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

#include "Multiforcer_CUDA_host/CHCommandLineData.h"
#include "Multiforcer_Common/CHCommon.h"
#include "Multiforcer_Common/CHHashes.h"

using namespace std;

// To set exit after found value
extern struct global_commands global_interface;


CHCommandLineData::CHCommandLineData() {
    // Set up some defaults.
    this->HashType = 0;
    this->UseCharsetMulti = 0;
    this->UseLookupTable = 0;
    this->UseOutputFile = 0;
    this->UseUnfoundOutputFile = 0;
    this->WorkunitBits = 0;
    this->UseRestoreFile = 0;

    // Set min/max password lengths.  Len16 is used because not all hashes
    // have longer options set.
    this->MinPasswordLength = 1;
    this->MaxPasswordLength = 0;

    this->CUDABlocks = 0;
    this->CUDAThreads = 0;
    this->TargetExecutionTimeMs = DEFAULT_CUDA_EXECUTION_TIME;
    this->UseZeroCopy = 0;

    this->CUDADevice = 0;
    this->CUDANumberDevices = 0;

    this->Autotune = 0;

    this->Silent = 0;
    this->Verbose = 0;
    this->Debug = 0;
    this->DevDebug = 0;
    this->Daemon = 0;

    this->IsNetworkClient = 0;
    this->IsNetworkServer = 0;
    this->NetworkPort = DEFAULT_NETWORK_PORT;
    this->IsServerOnly = 0;

    // Zero out the strings
    memset(this->HashListFileName, 0, sizeof(this->HashListFileName));
    memset(this->CharsetFileName, 0, sizeof(this->CharsetFileName));
    memset(this->OutputFileName, 0, sizeof(this->OutputFileName));
    memset(this->NetworkRemoteHost, 0, sizeof(this->NetworkRemoteHost));
    memset(this->UnfoundOutputFileName, 0, sizeof(this->UnfoundOutputFileName));
    memset(this->RestoreFileName, 0, sizeof(this->RestoreFileName));

    this->AddHexOutput = 0;

    this->CommandLineValid = 0;

}


CHCommandLineData::~CHCommandLineData() {

}

// Parses the command line.  Returns 0 for failure, 1 for success.
int CHCommandLineData::ParseCommandLine(int argc, char *argv[]) {
    int deviceCount, i;
    CHHashes HashTypes;
    int errors = 0;


    // Use types: Standalone, Server, Slave

    // h: Hash time, Required for Standalone, Server.
    struct arg_str *h = arg_str0("h", "hashtype", "{MD5, NTLM, LM, ...}", "type of hash to crack");
    // c: Charset files, require this or cm for Standalone, Server.
    struct arg_file *c = arg_file0("c", "charsetfile", "<file>", "charset file (non-deliniated)");
    struct arg_file *cm = arg_file0("u", "charsetfilemulti", "<file>", "charset file (newline between each line)");
    // o, n: Output files.  Not required, valid in all modes.
    struct arg_file *o = arg_file0("o", "outputfile", "<file>", "output file for results");
    struct arg_file *n = arg_file0("n", "notfoundfile", "<file>", "file for not found results");
    // h: Hashfile.  Required for Standalone, Server.
    struct arg_file *f = arg_file0("f", "hashfile", "<file>", "hash file (one per line, ASCII-hex)");
    // v: Verbose output.  Valid in all modes.
    struct arg_lit *v = arg_lit0("v", "verbose", "verbose output");
    // l: Use large lookup table.  Should be depreciated soon.  Valid in all modes.
    struct arg_lit *l = arg_lit0("l", "lookup", "BIG lookup table (useful for very long hash lists)");
    // min, max: Valid for Standalone, Server.  Defaults to 1, password_max_length.
    struct arg_int *min = arg_int0(NULL, "min", "<n>", "minimum password length");
    struct arg_int *max = arg_int0(NULL, "max", "<n>", "maximum password length");
    // d: CUDA device, if limited to one device.
    struct arg_int *d = arg_int0("d", "device", "<n>", "CUDA device to use");
    // m: Kernel time in ms.  Valid for all.
    struct arg_int *m = arg_int0("m", "ms", "<n>", "target step time in ms");
    // b, t: CUDA blocks/threads.  Valid for all.
    struct arg_int *b = arg_int0("b", "blocks", "<n>", "number of thread blocks to run");
    struct arg_int *t = arg_int0("t", "threads", "<n>", "number of threads per block");
    // mthreads: Max threads.  Not currently valid.
    struct arg_lit *mthreads = arg_lit0(NULL, "maxthreads", "use maximum number of threads possible");
    // autotune: Not currently valid.
    struct arg_lit *autotune = arg_lit0(NULL, "autotune", "autotune for maximum performance");
    // s: Silence ALL output except hashes.  Not working right now.  Valid in all modes.
    struct arg_lit *s = arg_lit0(NULL, "silent", "silence all output except passwords");
    // debug: Single GPU, verbose output.  Valid in all modes.
    struct arg_lit *debug = arg_lit0(NULL, "debug", "debug output mode");
    // devdebug: Single GPU, developer verbose output.  Valid in all modes.
    struct arg_lit *devdebug = arg_lit0(NULL, "devdebug", "developer debug output mode");
    // bits: Workunit bits.  Valid in Standalone, Server.
    struct arg_int *bits = arg_int0(NULL, "bits", "<n>", "workunit bits");
    // zero_copy: Force CUDA zerocopy memory.  Valid in all modes.
    struct arg_lit *zero_copy = arg_lit0(NULL, "zerocopy", "force CUDA zero copy");
    // network_server: Enable server.  Forces server mode.
    struct arg_lit *network_server = arg_lit0(NULL, "enableserver", "enable network server");
    // server_only: Run as a server with NO local GPU clients - should be more reliable.
    struct arg_lit *server_only = arg_lit0(NULL, "serveronly", "enable network server with no local compute");
    // network_port: Sets the network port to a non-default value.  Valid for server, client.
    struct arg_int *network_port = arg_int0("p", "port", "[port]", "port for network client or server");
    // network_client: Remote IP to connect to.  Valid for client only.
    struct arg_str *network_client = arg_str0(NULL, "remoteip","<hostname>", "remote network IP");
    // exit_after_found: Terminate after finding N passwords.  Valid for standalone, server.
    struct arg_int *exit_after_found = arg_int0(NULL, "onlyfind", "<n>", "max passwords to find");
    // hexoutput: Adds hex output to all password outputs.
    struct arg_lit *hex_output = arg_lit0(NULL, "hexoutput", "Adds hex output to all hash outputs");
    // Daemon - just sit there and don't output anything!
    struct arg_lit *daemon = arg_lit0(NULL, "daemon", "Client mode only: Sit and spin quietly.");
    // q: Device query.  Print all CUDA devices and exit.
    struct arg_lit *q = arg_lit0("q", "device-query", "print all CUDA devices found");
    // help: Some basic help
    struct arg_lit *help  = arg_lit0(NULL,"help", "print this help and exit");
    // restorefile: Restore state from this filename
    struct arg_file *restorefile = arg_file0(NULL, "resumefile", "<file>", "restore file: Resume previous cracking attempt");

    struct arg_end *end = arg_end(20);
    void *argtable[] = {h,c,cm,o,n,f,v,l,min,max,d,m,b,t,mthreads,autotune,s,debug,devdebug,
        exit_after_found,q,zero_copy,network_server,server_only,network_client, network_port,
        bits,help,hex_output,daemon,restorefile,end};


    if (arg_nullcheck(argtable) != 0)
      printf("error: insufficient memory\n");
    // Look for errors
    int nerrors = arg_parse(argc,argv,argtable);
    if (nerrors > 0) {
      // Print errors, exit.
      arg_print_errors(stdout,end,argv[0]);
      exit(1);
    }

    // The server-only version does not require CUDA cards.
    // Everything else does.  Verify this.
    if (!server_only->count) {
        // Check for a supported CUDA device & set the device count
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
          printf("This program requires a CUDA-capable video card.\nNo cards found.  Sorry.  Exiting.\n");
          printf("This is currently built against CUDA 3.2 and requires at least\n");
          printf("the version 260 drivers to work.  Try updating if you have a CUDA card.\n");
          exit(1);
        }
        this->CUDANumberDevices = deviceCount;
    }

    // If we are supposed to print the device info for all devices, do it.
    // This terminates after completion.
    if (q->count) {
        for (i = 0; i < this->CUDANumberDevices; i++) {
            printCudaDeviceInfo(i);
        }
        exit(0);
    }

    // Same for help.

    if (help->count) {
        printf("Usage: %s", argv[0]);
        arg_print_syntax(stdout,argtable,"\n\n");
        arg_print_glossary(stdout,argtable,"  %-10s %s\n");
        HashTypes.PrintAllHashTypes();
        exit(0);
    }

    // If a hash type is specified, make sure it is valid.
    if (h->count) {
        this->HashType = HashTypes.GetHashIdFromString(*h->sval);

        if (this->HashType == -1) {
          printf("Unknown hash type %s: Exiting.\n", *h->sval);
          exit(1);
        }
    }

    // Sanity check the input values for client/server/etc.

    if (network_client->count && (network_server->count || server_only->count)) {
        printf("Cannot be both network server and client!\n");
        exit(1);
    }

    if (s->count && v->count) {
        printf("Cannot specify both silent and verbose!\n");
        errors = 1;
    }

    if (restorefile->count) {
        // Check to make sure nothing restore-file specific is set.
        // Cannot set:
        // - Hashfile
        // - Charsetfile
        // - Output file
        // - Unfound file
        // - AddHexOutput
        // - HashType
        // - Password lengths

        if (f->count) {
            printf("Cannot set hash file when using restore file!\n");
            exit(1);
        }
        if (c->count || cm->count) {
            printf("Cannot set charset file when using restore file!\n");
            exit(1);
        }
        if (o->count) {
            printf("Cannot set output file when using restore file!\n");
            exit(1);
        }
        if (n->count) {
            printf("Cannot set unfound file when using restore file!\n");
            exit(1);
        }
        if (hex_output->count) {
            printf("Cannot set addhexoutput when using restore file!\n");
            exit(1);
        }
        if (h->count) {
            printf("Cannot set hash type when using restore file!\n");
            exit(1);
        }
        if (min->count || max->count) {
            printf("Cannot set min or max password length when using restore file!\n");
            exit(1);
        }

        this->UseRestoreFile = 1;
        strncpy(this->RestoreFileName, restorefile->filename[0], MAX_FILENAME_LENGTH - 1);

    } else if (!network_client->count) {
        // This is a standalone instance or a server.

        // Check for hash type
        if (!h->count) {
            printf("Hash type (-h) must be specified.\n");
            errors = 1;
        }
        // Check for charset
        if (!(c->count || cm->count)) {
            printf("Charset file (-c or -u) must be specified.\n");
            errors = 1;
        }
        if (c->count && cm->count) {
            printf("Must specify single charset (-c) or multi charset (-u), not both.\n");
            errors = 1;
        }
        if (!f->count) {
            printf("Hash file (-f ) must be specified.\n");
            errors = 1;
        }
        if (min->count && max->count && (*min->ival > *max->ival)) {
            printf("Minimum password length (%d) greater than maximum password length (%d).\n", *min->ival, *max->ival);
            errors = 1;
        }
        if (min->count && (*min->ival < 0)) {
          printf("Minimum password length (--min) must be greater than zero.\n");
          errors = 1;
        }
        if (max->count && (*max->ival > MAX_PASSWORD_LEN)) {
          printf("Maximum password length (--max) must be less than or equal to %d.\n", MAX_PASSWORD_LEN);
          errors = 1;
        }
    } else {
        // If this is a network client, they cannot specify certain things.
        if (h->count) {
            printf("Cannot specify hash type (-h) when network client.\n");
            errors = 1;
        }
        if (c->count || cm->count) {
            printf("Cannot specify charset (-c or -u) when network client.\n");
            errors = 1;
        }
        if (f->count) {
            printf("Cannot specify hash file (-f) when network client.\n");
            errors = 1;
        }
        // Daemon only works for client side.
        if (daemon->count) {
            this->Daemon = 1;
        }
    }

    if (errors) {
        printf("\nErrors in command line.  Exiting.\n");
        exit(1);
    }

    // Basic sanity tests passed, continue with setting values.

    // Basic modifiers
    if (v->count) {
      this->Verbose = 1;
      printf("Verbose mode enabled.\n");
    }
    if (s->count) {
        this->Silent = 1;
    }
    if (debug->count) {
        this->Debug = 1;
    }
    if (devdebug->count) {
        this->Debug = 1;
        this->DevDebug = 1;
    }

    // Network data
    if (network_server->count) {
        this->IsNetworkServer = 1;
    }

    if (server_only->count) {
        this->IsNetworkServer = 1;
        this->IsServerOnly = 1;
        this->CUDANumberDevices = 0;
    }
    
    if (network_client->count) {
        this->IsNetworkClient = 1;
        strncpy(this->NetworkRemoteHost, *network_client->sval, MAX_HOSTNAME_LENGTH);
    }

    if (network_port->count) {
        this->NetworkPort = *network_port->ival;
    }



    // Other hash related values.
    if (min->count) {
        this->MinPasswordLength = *min->ival;
    }
    if (max->count) {
        this->MaxPasswordLength = *max->ival;
    }

    if(bits->count) {
        this->WorkunitBits = *bits->ival;
    }

    if (c->count) {
        strncpy(this->CharsetFileName, c->filename[0], MAX_FILENAME_LENGTH - 1);
    }
    if(cm->count) {
        strncpy(this->CharsetFileName, cm->filename[0], MAX_FILENAME_LENGTH - 1);
        this->UseCharsetMulti = 1;
    }

    if (f->count) {
        strncpy(this->HashListFileName, f->filename[0], MAX_FILENAME_LENGTH - 1);
    }

    if (o->count) {
        strncpy(this->OutputFileName, o->filename[0], MAX_FILENAME_LENGTH - 1);
        this->UseOutputFile = 1;
    }
    if (n->count) {
        strncpy(this->UnfoundOutputFileName, n->filename[0], MAX_FILENAME_LENGTH - 1);
        this->UseUnfoundOutputFile = 1;
    }


    if (l->count) {
      this->UseLookupTable = 1;
    }
    if (b->count) {
      this->CUDABlocks = *b->ival;
    }
    if (t->count) {
      this->CUDAThreads = *t->ival;
    }
    if (zero_copy->count) {
        this->UseZeroCopy = 1;
    }

    // If we have an "exit after this many found" value, set it here.
    if (exit_after_found->count) {
        if (*exit_after_found->ival > 0) {
           global_interface.exit_after_count = *exit_after_found->ival;
        }
    }

    if (d->count) {
        if (*d->ival >= this->CUDANumberDevices) {
            printf("Device ID %d is not a valid device!\n", *d->ival);
            exit(1);
        }
        this->CUDADevice = *d->ival;
    }

    if (m->count) {
        this->TargetExecutionTimeMs = *m->ival;
    }
    if (hex_output->count) {
        this->AddHexOutput = 1;
    }

    if (autotune->count) {
      this->Autotune = 1;
    }

    return 1;
}



// Getters, all the setting is done in ParseCommandLine
int CHCommandLineData::GetHashType() {
    return this->HashType;
}

char* CHCommandLineData::GetHashListFileName() {
    return this->HashListFileName;
}

char* CHCommandLineData::GetCharsetFileName() {
    return this->CharsetFileName;
}

char CHCommandLineData::GetUseCharsetMulti() {
    return this->UseCharsetMulti;
}

char* CHCommandLineData::GetOutputFileName() {
    return this->OutputFileName;
}

char CHCommandLineData::GetUseOutputFile() {
    return this->UseOutputFile;
}

char* CHCommandLineData::GetUnfoundOutputFileName() {
    return this->UnfoundOutputFileName;
}

char CHCommandLineData::GetUseUnfoundOutputFile() {
    return this->UseUnfoundOutputFile;
}

int CHCommandLineData::GetCUDADevice() {
    return this->CUDADevice;
}


int CHCommandLineData::GetCUDANumberDevices() {
    return this->CUDANumberDevices;
}

int CHCommandLineData::GetTargetExecutionTimeMs() {
    return this->TargetExecutionTimeMs;
}

void CHCommandLineData::SetCUDABlocks(int CUDABlocks) {
    this->CUDABlocks = CUDABlocks;
}

int CHCommandLineData::GetCUDABlocks() {
    return this->CUDABlocks;
}

void CHCommandLineData::SetCUDAThreads(int CUDAThreads) {
    this->CUDAThreads = CUDAThreads;
}

int CHCommandLineData::GetCUDAThreads() {
    return this->CUDAThreads;
}

char CHCommandLineData::GetUseLookupTable() {
    return this->UseLookupTable;
}

char CHCommandLineData::GetAutotune() {
    return this->Autotune;
}

char CHCommandLineData::GetVerbose() {
    return this->Verbose;
}

char CHCommandLineData::GetSilent() {
    return this->Silent;
}

char CHCommandLineData::GetDebug() {
    return this->Debug;
}

char CHCommandLineData::GetDevDebug() {
    return this->DevDebug;
}

int CHCommandLineData::GetMinPasswordLength() {
    return this->MinPasswordLength;
}

int CHCommandLineData::GetMaxPasswordLength() {
    return this->MaxPasswordLength;
}

int CHCommandLineData::GetUseZeroCopy() {
    return this->UseZeroCopy;
}

void CHCommandLineData::SetUseZeroCopy(int newUseZeroCopy) {
    this->UseZeroCopy = newUseZeroCopy;
}
int CHCommandLineData::GetWorkunitBits() {
    return this->WorkunitBits;
}

char CHCommandLineData::GetIsNetworkServer() {
    return this->IsNetworkServer;
}
char CHCommandLineData::GetIsNetworkClient() {
    return this->IsNetworkClient;
}
char CHCommandLineData::GetIsServerOnly() {
    return this->IsServerOnly;
}
char *CHCommandLineData::GetNetworkRemoteHostname() {
    return this->NetworkRemoteHost;
}
uint16_t CHCommandLineData::GetNetworkPort() {
    return this->NetworkPort;
}
char CHCommandLineData::GetAddHexOutput() {
    return this->AddHexOutput;
}
char CHCommandLineData::GetDaemon() {
    return this->Daemon;
}
char* CHCommandLineData::GetRestoreFileName() {
    return this->RestoreFileName;
}
char CHCommandLineData::GetUseRestoreFile() {
    return this->UseRestoreFile;
}

std::vector<uint8_t> CHCommandLineData::GetRestoreData(int passLength) {
    CHSaveRestoreData RestoreData;
    std::vector<uint8_t> RestoreReturn;

    memset(&RestoreData, 0, sizeof(CHSaveRestoreData));

    RestoreData.CHSaveRestoreDataVersion = SAVE_RESTORE_DATA_VERSION;
    RestoreData.HashType = this->HashType;
    RestoreData.AddHexOutput = this->AddHexOutput;
    RestoreData.CurrentPasswordLength = passLength;
    RestoreData.UseCharsetMulti = this->UseCharsetMulti;

    strcpy(RestoreData.HashListFileName, this->HashListFileName);
    strcpy(RestoreData.CharsetFileName, this->CharsetFileName);
    strcpy(RestoreData.OutputFileName, this->OutputFileName);
    strcpy(RestoreData.UnfoundOutputFileName, this->UnfoundOutputFileName);

    RestoreReturn.resize(sizeof(CHSaveRestoreData));

    memcpy(&RestoreReturn[0], &RestoreData, sizeof(CHSaveRestoreData));

    return RestoreReturn;
}

void CHCommandLineData::SetDataFromRestore(std::vector<uint8_t> RestoreVector) {
    struct CHSaveRestoreData RestoreData;
    CHHashes HashTypes;

    if (RestoreVector.size() != sizeof(CHSaveRestoreData)) {
        printf("Restore data size incorrect!\n");
        printf("Need %u bytes, got %u bytes!\n",
               (unsigned int)sizeof(CHSaveRestoreData),
               (unsigned int)RestoreVector.size());
        exit(1);
    }

    memcpy(&RestoreData, &RestoreVector[0], sizeof(CHSaveRestoreData));

    // Do a sanity check on the restore version to make sure it matches.
    if (RestoreData.CHSaveRestoreDataVersion != SAVE_RESTORE_DATA_VERSION) {
        printf("Error: Restore metadata version does not match!\n");
        printf("Need version %d, got version %d\n", SAVE_RESTORE_DATA_VERSION, RestoreData.CHSaveRestoreDataVersion);
        printf("Cannot continue.\n");
        exit(1);
    }

    printf("Restoring data from resume file: Press ctrl-c if incorrect!\n");
    printf("Hash type: %s\n", HashTypes.GetHashStringFromID(RestoreData.HashType));
    printf("Hash filename: %s\n", RestoreData.HashListFileName);
    printf("Charset filename: %s\n", RestoreData.CharsetFileName);
    printf("Charset is multi-position: %s\n", RestoreData.UseCharsetMulti ? "Yes" : "No");
    printf("Password length: %d\n", RestoreData.CurrentPasswordLength);
    printf("Output filename: %s\n", RestoreData.OutputFileName);
    printf("Unfound filename: %s\n", RestoreData.UnfoundOutputFileName);
    printf("Hex password output: %s\n", RestoreData.AddHexOutput ? "Yes" : "No");

    printf("Please verify... waiting 10s...\n");
    CHSleep(10);


    this->HashType = RestoreData.HashType;

    // Restore is only valid for a single password length.
    this->MinPasswordLength = RestoreData.CurrentPasswordLength;
    this->MaxPasswordLength = RestoreData.CurrentPasswordLength;

    if (strlen(RestoreData.CharsetFileName)) {
        strcpy(this->CharsetFileName, RestoreData.CharsetFileName);
        this->UseCharsetMulti = RestoreData.UseCharsetMulti;
    } else {
        printf("No charset filename saved!  Cannot continue!\n");
        exit(1);
    }

    if (strlen(RestoreData.HashListFileName)) {
        strcpy(this->HashListFileName, RestoreData.HashListFileName);
    } else {
        printf("No hashlist filename saved!  Cannot continue!\n");
        exit(1);
    }

    if (strlen(RestoreData.OutputFileName)) {
        this->UseOutputFile = 1;
        strcpy(this->OutputFileName, RestoreData.OutputFileName);
        this->AddHexOutput = RestoreData.AddHexOutput;
    }

    if (strlen(RestoreData.UnfoundOutputFileName)) {
        this->UseUnfoundOutputFile = 1;
        strcpy(this->UnfoundOutputFileName, RestoreData.UnfoundOutputFileName);
    }
}

