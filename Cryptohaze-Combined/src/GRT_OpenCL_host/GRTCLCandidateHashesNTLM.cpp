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

// Pulls the .cl files into the kernel.
#define RELEASE_KERNEL 1

#include "GRT_OpenCL_host/GRTCLCandidateHashesNTLM.h"
#include <stdio.h>

#if RELEASE_KERNEL
extern char GRT_OpenCL_Common_source[];
extern char GRT_OpenCL_NTLM_source[];
extern char GRTCLCandidateHashesNTLM_source[];
#endif

GRTCLCandidateHashesNTLM::GRTCLCandidateHashesNTLM() : GRTCLCandidateHashes(16) {
    //printf("GRTCLCandidateHashesNTLM::GRTCLCandidateHashesNTLM()\n");
}

std::vector<std::string> GRTCLCandidateHashesNTLM::getHashFileName() {
    std::string HashFileName;
    std::vector<std::string> filesToReturn;

#if !RELEASE_KERNEL
    HashFileName = "kernels/GRT_OpenCL_Common.h";
    filesToReturn.push_back(HashFileName);
    HashFileName = "kernels/GRT_OpenCL_NTLM.h";
    filesToReturn.push_back(HashFileName);
    HashFileName = "kernels/GRTCLCandidateHashesNTLM.cl";
    filesToReturn.push_back(HashFileName);
#endif
    
    return filesToReturn;
}

std::string GRTCLCandidateHashesNTLM::getHashKernelName() {
    std::string HashKernelName;

    HashKernelName = "CandidateHashNTLMAMD";

    return HashKernelName;
}

std::string GRTCLCandidateHashesNTLM::getKernelSourceString() {
    std::string ReturnString;
    
#if RELEASE_KERNEL
    ReturnString += std::string(GRT_OpenCL_Common_source);
    ReturnString += std::string(GRT_OpenCL_NTLM_source);
    ReturnString += std::string(GRTCLCandidateHashesNTLM_source);
#endif
    
    return ReturnString;
}
