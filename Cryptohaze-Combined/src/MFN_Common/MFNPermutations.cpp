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


#include "MFN_Common/MFNPermutations.h"
#include "MFN_Common/MFNDebugging.h"
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>


MFNPermutations::MFNPermutations() {
    trace_printf("MFNPermutations::MFNPermutations()\n");
    // Initialize the permutation array with itsself - this is the default in
    // the case of no permutations.
    int i;
    this->loadedPermutations.clear();
    this->loadedPermutations.resize(256);
    for (i = 0; i < 256; i++) {
        this->loadedPermutations[i].push_back(i);
    }
}

MFNPermutations::~MFNPermutations() {
    trace_printf("MFNPermutations::~MFNPermutations()\n");
    // Clear the vector.  Probably don't need to explicitly do this.
    this->loadedPermutations.clear();
}

int MFNPermutations::LoadPermutations(std::string permutationFilename) {
    std::ifstream permutationFile;
    std::string fileLine;

    std::string whitespaces (" \t\f\v\n\r");
    size_t found;

    permutationFile.open(permutationFilename.c_str(), std::ios_base::in);
    if (!permutationFile.good())
    {

        std::cout << "ERROR: Cannot open hashfile " << permutationFilename <<"\n";
        exit(1);
    }

    while (std::getline(permutationFile, fileLine)) {
        found=fileLine.find_last_not_of(whitespaces);
        if (found!=std::string::npos)
            fileLine.erase(found+1);
        else
            fileLine.clear();
        printf("Line length: %d\n", (int)fileLine.length());

        // If the character at position 0 is > 255, something weird has happened.
        if (fileLine[0] > 255) {
            printf("Fatal error: Non-ASCII character detected!\n");
        }

        // If it's a valid line, do the work.
        if (fileLine.length() > 0) {
            for (int pos = 1; pos < fileLine.length(); pos++) {
                this->loadedPermutations[fileLine[0]].push_back(fileLine[pos]);
            }
        }
    }

    return 1;
}

#ifdef UNIT_TEST

int main(int argc, char *argv[]) {
    MFNPermutations Permutations;

    if (argc != 2) {
        printf("Error: Pass in permutation file.\n");
        exit(1);
    }

    printf("Trying to load permutations.\n");
    Permutations.LoadPermutations(std::string(argv[1]));
    printf("Permutations loaded.\n");

    for (int i = 0; i < 255; i++) {
        std::vector<uint8_t> charPerms;
        printf("%03d (%c) (%d perms): ", i, i,
                (int)Permutations.GetPermutationsForCharacter(i).size());
        charPerms = Permutations.GetPermutationsForCharacter(i);
        for (int j = 0; j < charPerms.size(); j++) {
            printf("%c ", charPerms[j]);
        }
        printf("\n");
    }
}

#endif
