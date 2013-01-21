/*
Cryptohaze GPU Rainbow Tables
Copyright (C) 2012  Bitweasil (http://www.cryptohaze.com/)

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

/**
 * This is an improved version of GRTTableVerify with more command line options.
 * 
 * It is designed to be run on directories full of files, and is multithreaded.
 * Note that it will not use more than one thread per table (the multithreaded
 * mode is useful for checking a directory full of files).
 * 
 * It verifies the table in the same way as the old binary, but adds the
 * following features:
 * 
 * --randompercent=%f : Check the specified percent of chains randomly through
 * the table.  This is useful for a quick check that things returned by webgen
 * are sane.
 * --sequential=%d : Check every d chains.  This is the same as the old
 * behavior, though it can check a large number of tables at once.
 * --movegood=%s : Move the good files to this directory.
 * --movebad=%s : Move the bad files to this directory.
 * --removebad : Just remove files that fail the check.
 * --showchains: Print all the chains being checked (requires single threaded).
 * --showsteps : Print all the steps being checked (requires single threaded).
 * --threads=%d : Number of threads to use.
 * 
 * Default behavior is to check all chains in all the specified tables with 1
 * thread.
 * 
 * Returns 0 if all tables pass, 1 if a table has failed.  Useful for scripted
 * use against one table.
 */

#include "GRT_Common/GRTTableHeaderV1.h"
#include "GRT_Common/GRTTableSearchV1.h"
#include "GRT_Common/GRTTableHeaderV2.h"
#include "GRT_Common/GRTTableSearchV2.h"
#include "GRT_Common/GRTTableHeaderV3.h"
#include "GRT_Common/GRTTableSearchV3.h"
#include "GRT_Common/GRTChainRunnerMD5.h"
#include "GRT_Common/GRTChainRunnerNTLM.h"
#include "GRT_Common/GRTChainRunnerSHA1.h"
#include "GRT_Common/GRTChainRunnerSHA256.h"

#include "CH_Common/CHRandom.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <argtable2.h>
#include <boost/filesystem.hpp>
#include <boost/thread.hpp>


int main(int argc, char *argv[]) {
    
    // First: Sort out the command line.
    struct arg_file *files_to_verify = arg_filen(NULL, NULL, "<table file>", 0, 100000, "GRT parts to verify.");
    struct arg_dbl *percent = arg_dbl0(NULL, "percent", "[0-100]", "Percent of chains to randomly verify.");
    struct arg_int *sequential = arg_int0(NULL, "sequential", "<n>", "Check every nth chain");
    /*
    void *argtable[] = {table_files, output_directory, bits, move_source, end};

    // Get arguments, collect data, check for basic errors.
    if (arg_nullcheck(argtable) != 0) {
      printf("error: insufficient memory\n");
    }
    // Look for errors
    int nerrors = arg_parse(argc,argv,argtable);
    if (nerrors > 0) {
      // Print errors, exit.
      arg_print_errors(stdout,end,argv[0]);
      // Print help.
      printf("\n\nOptions: \n");
      arg_print_glossary(stdout,argtable,"  %-20s %s\n");
      exit(1);
    }
    */
}
