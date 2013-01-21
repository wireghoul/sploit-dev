/*
Cryptohaze GPU Rainbow Tables
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

#include "GRT_OpenCL_host/GRTCLGenCommandLineData.h"
#include <stdint.h>
#include <argtable2.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "GRT_Common/GRTCommon.h"
#include "OpenCL_Common/GRTOpenCL.h"

GRTCLGenCommandLineData::GRTCLGenCommandLineData() {
    this->CommandLineValid = 0;
    this->GRTHashTypes = new GRTHashes;
    this->HashType = -1;
    // Default to version 2 output
    this->OutputTableVersion = 2;

    this->ChainLength = 0;

    // Default block/thread count
    this->OpenCLWorkitems = DEFAULT_WORKITEM_COUNT;
    this->OpenCLWorkgroups = DEFAULT_WORKGROUP_COUNT;

    this->OpenCLDevice = 0;
    this->OpenCLPlatform = 0;
    
    this->useAmdKernels = 0;
    this->vectorWidth = 4;

    this->OutputBits = DEFAULT_V3_BITS;

    this->RandomGenerator = NULL;

    // Default: Webgen to the default URL.
    this->useWebGenerate = 1;
    this->WebGenURL = std::string("http://webgen.cryptohaze.com/webgen.php");

    this->PasswordLength = 0;
    this->TableIndex = 0;
    this->NumberChains = 0;
    this->NumberTables = 0;
    this->ChainLength = 0;
    this->RandomSeed = 0;
}

GRTCLGenCommandLineData::~GRTCLGenCommandLineData() {

}

void GRTCLGenCommandLineData::PrintTableData() {
    // We print the table generate data here.
    printf("Table generation information:\n\n");
    printf("  Hash type: %s\n", this->GRTHashTypes->GetHashStringFromId(this->HashType));
    printf("  Password length: %d\n", this->PasswordLength);
    printf("  Table index: %u\n", this->TableIndex);
    printf("  Number of chains: %d\n", this->NumberChains);
    printf("  Number of tables: %d\n", this->NumberTables);
    printf("  Chain length: %d\n", this->ChainLength);
    printf("  Generate seed: %u\n", this->RandomSeed);
    printf("  Charset filename: %s\n", this->CharsetFileName);
    printf("\n\n");
}

// Parses the command line.  Returns 0 for failure, 1 for success.
int GRTCLGenCommandLineData::ParseCommandLine(int argc, char *argv[]) {
    if (this->RandomGenerator == NULL) {
        printf("Must set RandomGenerator before calling ParseCommandLine!\n");
        exit(1);
    }

    // Command line argument parsing with argtable
    struct arg_lit *verbose = arg_lit0("v", "verbose", "verbose output");

    // Table related options
    struct arg_str *hash_type = arg_str0("h", "hashtype", "{NTLM, MD4, MD5, SHA1}", "hash type to create tables for");
    struct arg_file *charset_file = arg_file0("c", "charsetfile", "charsetfile", "charset file");
    struct arg_int *pass_len = arg_int0("l", "passwordlen", "<n>", "password length to use");
    struct arg_int *table_index = arg_int0("i", "tableindex", "<n>", "table index");
    struct arg_int *chain_length = arg_int0(NULL, "chainlength", "<n>", "length of each chain");
    struct arg_int *num_chains = arg_int0(NULL, "numchains", "<n>", "number of chains in each table");
    struct arg_int *num_tables = arg_int0(NULL, "numtables", "<n>", "number of tables to create");
    struct arg_int *seed = arg_int0("s", "seed", "<n>", "Seed value to use");

    // CUDA related params
    struct arg_int *device = arg_int0("d", "device", "<n>", "OpenCL device to use");
    struct arg_int *platform = arg_int0("p", "platform", "<n>", "OpenCL platform to use");
    struct arg_int *m = arg_int0("m", "ms", "<n>", "target step time in ms");
    struct arg_int *blocks = arg_int0("b", "blocks", "<n>", "number of thread blocks to run");
    struct arg_int *threads = arg_int0("t", "threads", "<n>", "number of threads per block");
    struct arg_lit *autotune = arg_lit0(NULL, "autotune", "autotune for maximum performance");

    struct arg_int *table_version = arg_int0(NULL, "tableversion", "<n>", "table version (1 or 2)");
    struct arg_int *table_bits = arg_int0(NULL, "bits", "<n>", "bits in table output");

    struct arg_lit *amd_kernels = arg_lit0(NULL, "bfi_int", "use AMD bfi_int");
    struct arg_int *vector_width = arg_int0(NULL, "vectorwidth", "<n>", "vector width");

    struct arg_str *webgen_url = arg_str0(NULL, "webgenerateurl", "<URL>", "URL of the web table script");
    struct arg_str *webgen_username = arg_str0(NULL, "webgenerateusername", "<username>", "Username, if required, for the web table script");
    struct arg_str *webgen_password = arg_str0(NULL, "webgeneratepassword", "<password>", "Password, if required, for the web table script");

    struct arg_end *end = arg_end(20);
    void *argtable[] = {verbose,hash_type,charset_file,pass_len,table_index,
        chain_length,num_chains,num_tables,seed,device,platform,m,blocks,threads,autotune,
        table_version,amd_kernels,vector_width,table_bits, webgen_url,
        webgen_username, webgen_password,end};

    // Get arguments, collect data, check for basic errors.
    if (arg_nullcheck(argtable) != 0) {
      printf("error: insufficient memory\n");
    }
    // Look for errors
    int nerrors = arg_parse(argc,argv,argtable);
    if (nerrors > 0) {
      // Print errors, exit.
      arg_print_errors(stdout,end,argv[0]);
      exit(1);
    }

    // If hash type is set, and webgen is not being used, clear the webgen flag.
    if (hash_type->count && !webgen_url->count) {
        this->useWebGenerate = 0;
        this->WebGenURL = std::string();
    }
    
    // Web generate stuff
    if (webgen_url->count) {
        this->useWebGenerate = 1;
        this->WebGenURL = *webgen_url->sval;
    }
    if (webgen_username->count) {
        this->WebGenUsername = *webgen_username->sval;
    }
    if (webgen_password->count) {
        this->WebGenPassword = *webgen_password->sval;
    }

    // If webgen is NOT being used, verify that we have everything we need.
    if (!this->useWebGenerate) {
        // Set if there is an error.
        int nonWebGenError = 0;
        if (!hash_type->count) {
            printf("Error: missing option -h|--hashtype={NTLM, MD4, MD5, SHA1}\n");
            nonWebGenError = 1;
        }
        if (!charset_file->count) {
            printf("Error: missing option -c|--charsetfile=charsetfile\n");
            nonWebGenError = 1;
        }
        if (!pass_len->count) {
            printf("Error: missing option -l|--passwordlen=<n>\n");
            nonWebGenError = 1;
        }
        if (!table_index->count) {
            printf("Error: missing option -i|--tableindex=<n>\n");
            nonWebGenError = 1;
        }
        if (!chain_length->count) {
            printf("Error: missing option --chainlength=<n>\n");
            nonWebGenError = 1;
        }
        if (nonWebGenError) {
            exit(1);
        }
        
        // Now set stuff if we are not doing webgen.

        if (table_version->count) {
            if ((*table_version->ival < 1) || (*table_version->ival > 3)) {
                printf("Invalid table version %d!\n", table_version->ival);
                exit(1);
            }
            this->OutputTableVersion = *table_version->ival;
        }

        // Password length
        if (*pass_len->ival > MAX_PASSWORD_LENGTH || *pass_len->ival < MIN_PASSWORD_LENGTH) {
            printf("ERROR: Password length (%d) must be between %d and %d.\n", *pass_len->ival, MIN_PASSWORD_LENGTH, MAX_PASSWORD_LENGTH );
            exit(1);
        } else {
            this->PasswordLength = *pass_len->ival;
        }

        // Table index - any integer value is fine here.
        this->TableIndex = *table_index->ival;

        // Chain length - warn if it seems odd.
        if (chain_length->count) {
            if (*chain_length->ival < CHAIN_LENGTH_WARN_MIN || *chain_length->ival > CHAIN_LENGTH_WARN_MAX) {
                printf("WARNING: Chain length of %d seems out of a normal range.\n", *chain_length->ival);
            }
            this->ChainLength = *chain_length->ival;
        }
    }
    
    // Move on and start collecting values.

    // Webgen will determine type automatically.
    if (!this->useWebGenerate) {
        this->HashType = this->GRTHashTypes->GetHashIdFromString(*hash_type->sval);
        if (this->HashType == -1) {
            printf("Unknown hash type %s: Exiting.\n\n", *hash_type->sval);
            exit(1);
        }
    }


    if (table_bits->count) {
        this->OutputBits = *table_bits->ival;
    }

    // Number of chains
    this->NumberChains = *num_chains->ival;

    //
    if(num_tables->count) {
        this->NumberTables = *num_tables->ival;
    } else {
        this->NumberTables = 1;
    }

    // Random seed
    if (seed->count) {
        // If the seed is specified, seed the random generator with it.
        this->RandomSeed = *seed->ival;
        this->RandomGenerator->setSeed(this->RandomSeed);
    } else {
        // Seed the random number generator, then use this to seed itsself.
        //mt_goodseed();
        //this->RandomSeed = mt_llrand();
        // Else, just get the current (hardware random) seed.
        this->RandomSeed = this->RandomGenerator->getSeed();
    }
    // Either way, initialize the mtwister with it.
    //mt_seed32new(this->RandomSeed);

    // Do this to emulate CUDA behavior for now...
    // Threads - if not set, leave at default 0
    if (threads->count) {
        this->OpenCLWorkitems = *threads->ival;
    }

    // Blocks - if not set, leave at default 0
    if (blocks->count) {
        this->OpenCLWorkgroups = *blocks->ival * this->OpenCLWorkitems;
    }


    // Autotune the news
    if (autotune->count) {
        this->Autotune = 1;
    } else {
        this->Autotune = 0;
    }

    strcpy((char *)this->CharsetFileName, (char *)charset_file->filename[0]);

    // Finally, set the CUDA device and look for errors.
    if (device->count) {
        this->OpenCLDevice = *device->ival;
    }
    if (platform->count) {
        this->OpenCLPlatform = *platform->ival;
    }

    // Desired kernel time - if not specified, set sanely based on timeouts.
    if (m->count) {
        this->KernelTimeMs = *m->ival;
    } else {
        this->KernelTimeMs = DEFAULT_KERNEL_TIME;
    }

    if (amd_kernels->count) {
        this->useAmdKernels = 1;
        this->vectorWidth = 4;
    }

    if (vector_width->count) {
        this->vectorWidth = *vector_width->ival;
    }

    return 1;
}

// Getters, all the setting is done in ParseCommandLine
int GRTCLGenCommandLineData::getHashType() {
    return this->HashType;
}
int GRTCLGenCommandLineData::getPasswordLength() {
    return this->PasswordLength;
}
uint32_t GRTCLGenCommandLineData::getTableIndex() {
    return this->TableIndex;
}
uint32_t GRTCLGenCommandLineData::getChainLength() {
    return this->ChainLength;
}
uint32_t GRTCLGenCommandLineData::getNumberChains() {
    return this->NumberChains;
}
void GRTCLGenCommandLineData::setNumberChains(uint32_t newNumberChains) {
    this->NumberChains = newNumberChains;
}

uint32_t GRTCLGenCommandLineData::getNumberTables() {
    return this->NumberTables;
}
uint32_t GRTCLGenCommandLineData::getRandomSeed() {
    return this->RandomSeed;
}
char* GRTCLGenCommandLineData::GetCharsetFileName() {
    return this->CharsetFileName;
}
unsigned int GRTCLGenCommandLineData::getOpenCLDevice() {
    return this->OpenCLDevice;
}
unsigned int GRTCLGenCommandLineData::getOpenCLPlatform() {
    return this->OpenCLPlatform;
}
unsigned int GRTCLGenCommandLineData::getKernelTimeMs() {
    return this->KernelTimeMs;
}
unsigned int GRTCLGenCommandLineData::getWorkgroups() {
    return this->OpenCLWorkgroups;
}
unsigned int GRTCLGenCommandLineData::getWorkitems() {
    return this->OpenCLWorkitems;
}
int GRTCLGenCommandLineData::getOutputTableVersion() {
    return this->OutputTableVersion;
}
int GRTCLGenCommandLineData::getUseAmdKernels() {
    return this->useAmdKernels;
}
int GRTCLGenCommandLineData::getVectorWidth() {
    return this->vectorWidth;
}
int GRTCLGenCommandLineData::getOutputTableBits() {
    return this->OutputBits;
}
