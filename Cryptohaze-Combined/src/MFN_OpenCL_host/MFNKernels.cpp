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
#include "MFN_OpenCL_device/MFN_OpenCL_Common_source.h"
#include "MFN_OpenCL_device/MFN_OpenCL_MD5_source.h"
#include "MFN_OpenCL_device/MFN_OpenCL_SHA256_source.h"
#include "MFN_OpenCL_device/MFN_OpenCL_NTLM_source.h"
#include "MFN_OpenCL_device/MFN_OpenCL_PasswordCopiers_source.h"
#include "MFN_OpenCL_device/MFN_OpenCL_SaltUtilities_source.h"
#include "MFN_OpenCL_device/MFN_OpenCL_BIN2HEX_source.h"


#include "MFN_OpenCL_device/MFNHashTypePlainOpenCL_16HEX_source.h"
#include "MFN_OpenCL_device/MFNHashTypePlainOpenCL_DoubleMD5_source.h"
#include "MFN_OpenCL_device/MFNHashTypePlainOpenCL_DupMD5_source.h"
#include "MFN_OpenCL_device/MFNHashTypePlainOpenCL_LOTUS_source.h"
#include "MFN_OpenCL_device/MFNHashTypePlainOpenCL_MD5WL_source.h"
#include "MFN_OpenCL_device/MFNHashTypePlainOpenCL_MD5_source.h"
#include "MFN_OpenCL_device/MFNHashTypePlainOpenCL_NTLM_source.h"
#include "MFN_OpenCL_device/MFNHashTypePlainOpenCL_NTLMWL_source.h"
#include "MFN_OpenCL_device/MFNHashTypePlainOpenCL_SNTLM_source.h"
#include "MFN_OpenCL_device/MFNHashTypePlainOpenCL_SHA1_source.h"
#include "MFN_OpenCL_device/MFNHashTypePlainOpenCL_SHA256_source.h"
#include "MFN_OpenCL_device/MFNHashTypePlainOpenCL_DoubleSHA256_source.h"
#include "MFN_OpenCL_device/MFNHashTypePlainOpenCL_SMD5_source.h"
#include "MFN_OpenCL_device/MFNHashTypeSaltedOpenCL_IPB_source.h"
#include "MFN_OpenCL_device/MFNHashTypeSaltedOpenCL_IPBWL_source.h"
#include "MFN_OpenCL_device/MFNHashTypeSaltedOpenCL_MD5_PS_source.h"
#include "MFN_OpenCL_device/MFNHashTypeSaltedOpenCL_Phpass_source.h"
#include "MFN_OpenCL_device/MFNHashTypeSaltedOpenCL_PhpassWL_source.h"

