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

// Shows the OpenCL platforms/devices

#include "OpenCL_Common/GRTOpenCL.h"
#include <stdlib.h>
#include <stdio.h>



int main(int argc, char *argv[]) {
    CryptohazeOpenCL *OpenCL, *OpenCLDevicePrintout;

    int i;

    OpenCL = new CryptohazeOpenCL();

    OpenCL->printAvailablePlatforms();

    for (i = 0; i < OpenCL->getNumberOfPlatforms(); i++) {
        printf("Platform %d: \n", i);
        OpenCLDevicePrintout = new CryptohazeOpenCL();
        OpenCLDevicePrintout->selectPlatformById(i);
        OpenCLDevicePrintout->printAvailableDevices();
        delete OpenCLDevicePrintout;
    }
}

