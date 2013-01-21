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

/**
 * This file does nothing but include the OpenCL generated source files.
 * It's to avoid duplication and name conflicts.
 */

// General includes.
#include "GRT_OpenCL_device/GRT_OpenCL_Common_source.h"

// Algorithm includes.
#include "GRT_OpenCL_device/GRT_OpenCL_MD5_source.h"
#include "GRT_OpenCL_device/GRT_OpenCL_NTLM_source.h"
#include "GRT_OpenCL_device/GRT_OpenCL_SHA1_source.h"
#include "GRT_OpenCL_device/GRT_OpenCL_SHA256_source.h"

// Source file includes
#include "GRT_OpenCL_device/GRTCLGenerateTableMD5_AMD_source.h"
#include "GRT_OpenCL_device/GRTCLGenerateTableNTLM_AMD_source.h"
#include "GRT_OpenCL_device/GRTCLGenerateTableSHA1_AMD_source.h"
#include "GRT_OpenCL_device/GRTCLGenerateTableSHA256_AMD_source.h"

#include "GRT_OpenCL_device/GRTCLCandidateHashesMD5_source.h"
#include "GRT_OpenCL_device/GRTCLCandidateHashesNTLM_source.h"
#include "GRT_OpenCL_device/GRTCLCandidateHashesSHA1_source.h"
#include "GRT_OpenCL_device/GRTCLCandidateHashesSHA256_source.h"

#include "GRT_OpenCL_device/GRTCLRegenerateChainsMD5_source.h"
#include "GRT_OpenCL_device/GRTCLRegenerateChainsNTLM_source.h"
#include "GRT_OpenCL_device/GRTCLRegenerateChainsSHA1_source.h"
#include "GRT_OpenCL_device/GRTCLRegenerateChainsSHA256_source.h"
