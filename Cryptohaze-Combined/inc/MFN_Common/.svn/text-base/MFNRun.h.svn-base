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

#ifndef __MFN_RUN_H__
#define __MFN_RUN_H__

#include "MFN_Common/MFNMultiforcerClassFactory.h"
#include "Multiforcer_Common/CHCommon.h"

// global_commands is a way of communicating across all threads.
// It handles exit requests and error handling.
extern struct global_commands global_interface;

/**
 * Global class factory.
 */
extern MFNClassFactory MultiforcerGlobalClassFactory;

// Runs based on the current settings
void MFNRun();

// Runs the multiforcer in standalone or network server mode.
void runStandaloneOrServerMode();

void runNetworkClientMode();

#endif