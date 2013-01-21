// OpenCL utilities for the Cryptohaze tool set

#ifndef __GRTOPENCL_H
#define __GRTOPENCL_H

#ifdef __APPLE_CC__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <stdint.h>
#include <stdio.h>

#include <vector>
#include <string>

#define MAX_SUPPORTED_PLATFORMS 16
#define MAX_SUPPORTED_DEVICES 16


// Platform defines.
#define PLATFORM_NVIDIA 0
#define PLATFORM_ATI 1
#define PLATFORM_INTEL 2
#define PLATFORM_APPLE 3
#define PLATFORM_OTHER 4

// Device defines
#define DEVICE_CPU 0
#define DEVICE_GPU 1
#define DEVICE_ALL 2




class CryptohazeOpenCL {
public:
    CryptohazeOpenCL();
    ~CryptohazeOpenCL();

    // Prints all available platforms to use.
    void printAvailablePlatforms();
    
    // Gets the number of platforms
    int getNumberOfPlatforms();
    
    // Choose a platform based on the ID (0 for default).  Returns true or false.
    void selectPlatformById(int);

    // Print out all available devices with some details
    void printAvailableDevices();

    // Gets the number of devices on the platform
    int getNumberOfDevices();
    
    // Set the device types to search for - DEVICE_CPU, DEVICE_GPU, DEVICE_ALL
    void setDeviceType(uint32_t);

    // Select a device to use.  Returns true for success, false for failure.
    void selectDeviceById(int);

    // Get a context to use
    void createContext();

    // Return the current context
    cl_context getContext();

    // Build an OpenCL program from a single source file
    void buildProgramFromSource(const char *filename, const char *options);

    // Build an OpenCL program from multiple source files
    void buildProgramFromManySources(std::vector<std::string>, const char *options);

    // Build an OpenCL program from multiple source files concat'd
    void buildProgramFromManySourcesConcat(std::vector<std::string>, 
        const char *options, std::string prependDefines = std::string());


    void doAMDBFIPatch();

    cl_program getProgram();

    void createCommandQueue();

    cl_command_queue getCommandQueue();

    // Returns the maximum alloc size for a device
    cl_ulong getMaximumAllocSize();
    
    // Returns the available global memory for a device
    cl_ulong getGlobalMemorySize();

    // Returns the available local (shared) memory for a device
    cl_ulong getLocalMemorySize();

    int getDefaultThreadCount();
    int getDefaultBlockCount();
    
    void enableDumpSourceFile() {
        this->dumpSourceFile = 1;
    }



private:
    // Populates the platforms, if not set.
    void populatePlatforms();
    // Set to 0 if platforms are not populated, else 1.
    char platformsPopulated;
    cl_platform_id OpenCLPlatforms[MAX_SUPPORTED_PLATFORMS];
    cl_uint numPlatformsPopulated;
    cl_uint maxPlatformsSupported;
    int currentPlatform;

    // Populate the devices, if not set.
    void populateDevices();
    char devicesPopulated;
    cl_device_id OpenCLDevices[MAX_SUPPORTED_DEVICES];
    cl_uint numDevicesPopulated;
    cl_uint maxDevicesSupported;
    int currentDevice;
    
    // Context related values
    cl_context OpenCLContext;
    char contextPopulated;

    // Program related values
    cl_program OpenCLProgram;
    char programPopulated;

    // Command queue
    cl_command_queue OpenCLCommandQueue;
    char commandQueuePopulated;

    int defaultThreadCount;
    int defaultBlockCount;
    
    uint32_t deviceMask;
    
    // If true, the OpenCL source file is dumped.
    char dumpSourceFile;
    
};

#endif
