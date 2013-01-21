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

#include "MFN_Common/MFNCommandLineData.h"
#include "Multiforcer_Common/CHCommon.h"
#include "Multiforcer_Common/CHHashes.h"

#include "MFN_Common/MFNNetworkCommon.h"

// For hardware concurrency data
#include <boost/thread.hpp>

// For CUDA device count data
#include "cuda.h"

// For OpenCL device count
#include "OpenCL_Common/GRTOpenCL.h"

// To set exit after found value
extern struct global_commands global_interface;


MFNCommandLineData::MFNCommandLineData() {
    // Set up some defaults.
    this->HashType = 0;
    this->UseCharsetMulti = 0;
    this->UseLookupTable = 0;
    this->WorkunitBits = 0;

    // Set min/max password lengths.  Len16 is used because not all hashes
    // have longer options set.
    this->MinPasswordLength = 1;
    this->MaxPasswordLength = 0;

    this->DefaultCUDABlocks = 0;
    this->DefaultCUDAThreads = 0;
    this->TargetExecutionTimeMs = 0;
    this->UseZeroCopy = 0;
    this->UseBFIIntPatching = 0;
    this->VectorWidth = 0;
    this->SharedBitmapSize = 0;

    this->Silent = 0;
    this->Verbose = 0;
    this->Debug = 0;
    this->DevDebug = 0;
    this->Daemon = 0;

    this->IsNetworkClient = 0;
    this->IsNetworkServer = 0;
    this->NetworkPort = MFN_NETWORK_DEFAULT_PORT;
    this->IsServerOnly = 0;

    this->AddHexOutput = 0;
    this->PrintAlgorithms = 0;
}


MFNCommandLineData::~MFNCommandLineData() {

}

// Parses the command line.  Returns 0 for failure, 1 for success.
int MFNCommandLineData::ParseCommandLine(int argc, char *argv[]) {
    int deviceCount, i;
    //CHHashes HashTypes;
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
    //struct arg_int *d = arg_int0("d", "device", "<n>", "CUDA device to use");
    // m: Kernel time in ms.  Valid for all.
    struct arg_int *m = arg_int0("m", "ms", "<n>", "target step time in ms");
    // b, t: CUDA blocks/threads.  Valid for all.
    struct arg_int *b = arg_int0("b", "blocks", "<n>", "number of thread blocks to run");
    struct arg_int *t = arg_int0("t", "threads", "<n>", "number of threads per block");
    // mthreads: Max threads.  Not currently valid.
    struct arg_lit *mthreads = arg_lit0(NULL, "maxthreads", "use maximum number of threads possible");
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
    // printalgorithm: Prints the algorithm type used for the hash
    struct arg_lit *printalgorithm = arg_lit0(NULL, "printalgorithm", "Adds the hash algorithm to the output file.");
    // Daemon - just sit there and don't output anything!
    struct arg_lit *daemon = arg_lit0(NULL, "daemon", "Client mode only: Sit and spin quietly.");
    // help: Some basic help
    struct arg_lit *help  = arg_lit0(NULL,"help", "print this help and exit");
    // restorefile: Restore state from this filename
    struct arg_file *restorefile = arg_file0(NULL, "resumefile", "<file>", "restore file: Resume previous cracking attempt");

    // Device identification & enumeration
    // CPU thread count.
    struct arg_int *cpu_threads = arg_int0(NULL, "cputhreads", "<int>", "Number of CPU threads to use");
    // CUDA device identifiers
    struct arg_int *cuda_devices = arg_intn("d", "cudadevice", "<int>", 0, 255, "CUDA device IDs (multiple allowed)");
    // OpenCL device identifiers
    struct arg_int *opencl_platform = arg_int0(NULL, "openclplatform", "<int>", "OpenCL platform to use");
    struct arg_int *opencl_devices = arg_intn(NULL, "opencldevice", "<int>", 0, 255, "OpenCL device IDs (multiple allowed");
    
    struct arg_lit *no_cuda = arg_lit0(NULL, "nocuda", "Do not use any CUDA threads.");
    struct arg_lit *no_opencl = arg_lit0(NULL, "noopencl", "Do not use any OpenCL threads.");
    struct arg_lit *no_cpu = arg_lit0(NULL, "nocpu", "Do not use any CPU threads.");
    struct arg_lit *bfi_int = arg_lit0(NULL, "bfi_int", "Use AMD BFI_INT patching");
    struct arg_int *vector_width = arg_int0(NULL, "vectorwidth", "<n>", "vector width");
    struct arg_int *shared_bitmap = arg_int0(NULL, "sharedbitmap", "<n>", "shared bitmap size (8, 16, 32)");
    
    struct arg_end *end = arg_end(20);
    void *argtable[] = {h,c,cm,o,n,f,v,l,min,max,m,b,t,mthreads,s,debug,devdebug,
        exit_after_found,zero_copy,network_server,server_only,network_client, network_port,
        bits,help,hex_output,daemon,restorefile,
        cpu_threads, cuda_devices, opencl_platform, opencl_devices,
        no_cuda, no_opencl, no_cpu, bfi_int, vector_width, printalgorithm,
        shared_bitmap, end};


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
        /*
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
          printf("This program requires a CUDA-capable video card.\nNo cards found.  Sorry.  Exiting.\n");
          printf("This is currently built against CUDA 3.2 and requires at least\n");
          printf("the version 260 drivers to work.  Try updating if you have a CUDA card.\n");
          exit(1);
        }
        this->CUDANumberDevices = deviceCount;
        */
    }

    // If we are supposed to print the device info for all devices, do it.
    // This terminates after completion.

    // Same for help.

    if (help->count) {
        printf("Usage: %s", argv[0]);
        arg_print_syntax(stdout,argtable,"\n\n");
        arg_print_glossary(stdout,argtable,"  %-10s %s\n");
        //HashTypes.PrintAllHashTypes();
        exit(0);
    }

    // If a hash type is specified, make sure it is valid.
    if (h->count) {
        //this->HashType = HashTypes.GetHashIdFromString(*h->sval);
        this->HashTypeString = h->sval[0];

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

        this->RestoreFileName = restorefile->filename[0];

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
    if (daemon->count) {
        this->Daemon = 1;
    }

    // Network data
    if (network_server->count) {
        this->IsNetworkServer = 1;
    }

    if (server_only->count) {
        this->IsNetworkServer = 1;
        this->IsServerOnly = 1;
    }
    
    if (network_client->count) {
        this->IsNetworkClient = 1;
        this->NetworkRemoteHost = network_client->sval[0];
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
        this->CharsetFileName = c->filename[0];
    }
    if(cm->count) {
        this->CharsetFileName = cm->filename[0];
        this->UseCharsetMulti = 1;
    }

    if (f->count) {
        this->HashListFileName = f->filename[0];
    }

    if (o->count) {
        this->OutputFileName = o->filename[0];
    }
    if (n->count) {
        this->UnfoundOutputFileName = n->filename[0];
    }


    if (l->count) {
      this->UseLookupTable = 1;
    }
    if (b->count) {
      this->DefaultCUDABlocks = *b->ival;
    }
    if (t->count) {
      this->DefaultCUDAThreads = *t->ival;
    }
    if (zero_copy->count) {
        this->UseZeroCopy = 1;
    }
    if (bfi_int->count) {
        this->UseBFIIntPatching = 1;
    }
    if (vector_width->count) {
        this->VectorWidth = *vector_width->ival;
    }
    if (shared_bitmap->count) {
        this->SharedBitmapSize = *shared_bitmap->ival;
    }

    // If we have an "exit after this many found" value, set it here.
    if (exit_after_found->count) {
        if (*exit_after_found->ival > 0) {
           global_interface.exit_after_count = *exit_after_found->ival;
        }
    }

    if (m->count) {
        this->TargetExecutionTimeMs = *m->ival;
    }
    if (hex_output->count) {
        this->AddHexOutput = 1;
    }
    if (printalgorithm->count) {
        this->PrintAlgorithms = 1;
    }
    
    // Handle devices sanely.
    {
        MFNDeviceInformation DeviceInfo;
        uint32_t OpenCL_Platform = 0;
        uint32_t OpenCL_Thread_Count = 0;
        uint32_t CUDA_Thread_Count = 0;
        int numberCudaDevices = 0;
        int device;
        
        // Handle CUDA threads
        if (!no_cuda->count) {
            memset(&DeviceInfo, 0, sizeof(MFNDeviceInformation));
            DeviceInfo.IsCUDADevice = 1;
            cudaGetDeviceCount(&numberCudaDevices);
            if (this->Debug || this->DevDebug) {
                    printf("Got %d CUDA devices!\n", numberCudaDevices);
            }
            if (cuda_devices->count) {
                // If CUDA devices are specified, add the specified ones.
                for (device = 0; device < cuda_devices->count; device++) {
                    DeviceInfo.GPUDeviceId = cuda_devices->ival[device];
                    this->DevicesToUse.push_back(DeviceInfo);
                    CUDA_Thread_Count++;
                }
            } else {
                // Just use them all.
                for (device = 0; device < numberCudaDevices; device++) {
                    DeviceInfo.GPUDeviceId = device;
                    this->DevicesToUse.push_back(DeviceInfo);
                    CUDA_Thread_Count++;
                }
            }
        }
        
        // OpenCL Devices
        if (!no_opencl->count) {
            memset(&DeviceInfo, 0, sizeof(MFNDeviceInformation));
            DeviceInfo.IsOpenCLDevice = 1;
            // If the OpenCL platform is specified, use it instead of the default 0.
            if (opencl_platform->count) {
                OpenCL_Platform = opencl_platform->ival[0];
            }
            DeviceInfo.OpenCLPlatformId = OpenCL_Platform;
            if (opencl_devices->count) {
                // If the devices are listed, use them.
                for (device = 0; device < opencl_devices->count; device++) {
                    DeviceInfo.GPUDeviceId = opencl_devices->ival[device];
                    this->DevicesToUse.push_back(DeviceInfo);
                    OpenCL_Thread_Count++;
                }
            } else {
                // Enumerate the OpenCL devices & add them all.
                CryptohazeOpenCL OpenCL;
                int numberOpenCLGPUs;

                OpenCL.getNumberOfPlatforms();
                OpenCL.selectPlatformById(OpenCL_Platform);
                OpenCL.setDeviceType(DEVICE_ALL);
                numberOpenCLGPUs = OpenCL.getNumberOfDevices();
                if (this->Debug || this->DevDebug) {
                    printf("Got %d OpenCL GPUs\n", numberOpenCLGPUs);
                }
                for (device = 0; device < numberOpenCLGPUs; device++) {
                    DeviceInfo.GPUDeviceId = device;
                    this->DevicesToUse.push_back(DeviceInfo);
                    OpenCL_Thread_Count++;
                }
            }            
        }

        
        if (!no_cpu->count) {
            memset(&DeviceInfo, 0, sizeof(MFNDeviceInformation));
            if (this->Debug || this->DevDebug) {
                printf("Got %d CPU cores\n", (int)boost::thread::hardware_concurrency());
            }
            if (cpu_threads->count) {
                // If the CPU thread count is specified, use it.
                if (cpu_threads->ival[0]) {
                    DeviceInfo.IsCPUDevice = 1;
                    DeviceInfo.DeviceThreads = cpu_threads->ival[0];
                    this->DevicesToUse.push_back(DeviceInfo);
                }
            } else {
                DeviceInfo.IsCPUDevice = 1;
                // If there are cores left, add them.
                if (((int)boost::thread::hardware_concurrency() 
                        - (int)OpenCL_Thread_Count - (int)CUDA_Thread_Count) > 0) {
                    DeviceInfo.DeviceThreads = (int)boost::thread::hardware_concurrency() 
                            - (int)OpenCL_Thread_Count - (int)CUDA_Thread_Count;
                    this->DevicesToUse.push_back(DeviceInfo);
                }
            }
        }
    }
    
    if (this->Debug || this->DevDebug) {
        for (int i = 0; i < this->DevicesToUse.size(); i++) {
            printf("Device %d: ", i);
            if (this->DevicesToUse[i].IsCPUDevice) {
                printf("CPU, threads: %d\n", this->DevicesToUse[i].DeviceThreads);
            } else if (this->DevicesToUse[i].IsCUDADevice) {
                printf("CUDA, device %d\n", this->DevicesToUse[i].GPUDeviceId);
            } else if (this->DevicesToUse[i].IsOpenCLDevice) {
                printf("OpenCL, p:%d, d:%d\n", this->DevicesToUse[i].OpenCLPlatformId,
                        this->DevicesToUse[i].GPUDeviceId);
            } else {
                printf("Unknown device!\n");
            }
        }
    }
    return 1;
}




std::vector<uint8_t> MFNCommandLineData::GetRestoreData(int passLength) {
    MFNSaveRestoreData RestoreData;
    std::vector<uint8_t> RestoreReturn;

    memset(&RestoreData, 0, sizeof(MFNSaveRestoreData));

    RestoreData.MFNSaveRestoreDataVersion = SAVE_RESTORE_DATA_VERSION;
    RestoreData.HashType = this->HashType;
    RestoreData.AddHexOutput = this->AddHexOutput;
    RestoreData.CurrentPasswordLength = passLength;
    RestoreData.UseCharsetMulti = this->UseCharsetMulti;

    strcpy(RestoreData.HashListFileName, this->HashListFileName.c_str());
    strcpy(RestoreData.CharsetFileName, this->CharsetFileName.c_str());
    strcpy(RestoreData.OutputFileName, this->OutputFileName.c_str());
    strcpy(RestoreData.UnfoundOutputFileName, this->UnfoundOutputFileName.c_str());

    RestoreReturn.resize(sizeof(MFNSaveRestoreData));

    memcpy(&RestoreReturn[0], &RestoreData, sizeof(MFNSaveRestoreData));

    return RestoreReturn;
}

void MFNCommandLineData::SetDataFromRestore(std::vector<uint8_t> RestoreVector) {
    struct MFNSaveRestoreData RestoreData;
    //CHHashes HashTypes;

    if (RestoreVector.size() != sizeof(MFNSaveRestoreData)) {
        printf("Restore data size incorrect!\n");
        printf("Need %d bytes, got %d bytes!\n", sizeof(MFNSaveRestoreData), RestoreVector.size());
        exit(1);
    }

    memcpy(&RestoreData, &RestoreVector[0], sizeof(MFNSaveRestoreData));

    // Do a sanity check on the restore version to make sure it matches.
    if (RestoreData.MFNSaveRestoreDataVersion != SAVE_RESTORE_DATA_VERSION) {
        printf("Error: Restore metadata version does not match!\n");
        printf("Need version %d, got version %d\n", SAVE_RESTORE_DATA_VERSION, RestoreData.MFNSaveRestoreDataVersion);
        printf("Cannot continue.\n");
        exit(1);
    }

    printf("Restoring data from resume file: Press ctrl-c if incorrect!\n");
    //printf("Hash type: %s\n", HashTypes.GetHashStringFromID(RestoreData.HashType));
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
        this->CharsetFileName = RestoreData.CharsetFileName;
        this->UseCharsetMulti = RestoreData.UseCharsetMulti;
    } else {
        printf("No charset filename saved!  Cannot continue!\n");
        exit(1);
    }

    if (strlen(RestoreData.HashListFileName)) {
        this->HashListFileName = RestoreData.HashListFileName;
    } else {
        printf("No hashlist filename saved!  Cannot continue!\n");
        exit(1);
    }

    if (strlen(RestoreData.OutputFileName)) {
        this->OutputFileName = RestoreData.OutputFileName;
        this->AddHexOutput = RestoreData.AddHexOutput;
    }

    if (strlen(RestoreData.UnfoundOutputFileName)) {
        this->UnfoundOutputFileName = RestoreData.UnfoundOutputFileName;
    }
}

