

#include "OpenCL_Common/GRTOpenCL.h"
#include <string.h>
#include "GRT_OpenCL_host/GRTCLUtils.h"
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>

#define UNIT_TEST 0

using namespace std;

#include "OpenCL_Common/BFIPatcher.h"


char *print_cl_errstring(cl_int err) {
    switch (err) {
        case CL_SUCCESS:                          return strdup("Success!");
        case CL_DEVICE_NOT_FOUND:                 return strdup("Device not found.");
        case CL_DEVICE_NOT_AVAILABLE:             return strdup("Device not available");
        case CL_COMPILER_NOT_AVAILABLE:           return strdup("Compiler not available");
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return strdup("Memory object allocation failure");
        case CL_OUT_OF_RESOURCES:                 return strdup("Out of resources");
        case CL_OUT_OF_HOST_MEMORY:               return strdup("Out of host memory");
        case CL_PROFILING_INFO_NOT_AVAILABLE:     return strdup("Profiling information not available");
        case CL_MEM_COPY_OVERLAP:                 return strdup("Memory copy overlap");
        case CL_IMAGE_FORMAT_MISMATCH:            return strdup("Image format mismatch");
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return strdup("Image format not supported");
        case CL_BUILD_PROGRAM_FAILURE:            return strdup("Program build failure");
        case CL_MAP_FAILURE:                      return strdup("Map failure");
        case CL_INVALID_VALUE:                    return strdup("Invalid value");
        case CL_INVALID_DEVICE_TYPE:              return strdup("Invalid device type");
        case CL_INVALID_PLATFORM:                 return strdup("Invalid platform");
        case CL_INVALID_DEVICE:                   return strdup("Invalid device");
        case CL_INVALID_CONTEXT:                  return strdup("Invalid context");
        case CL_INVALID_QUEUE_PROPERTIES:         return strdup("Invalid queue properties");
        case CL_INVALID_COMMAND_QUEUE:            return strdup("Invalid command queue");
        case CL_INVALID_HOST_PTR:                 return strdup("Invalid host pointer");
        case CL_INVALID_MEM_OBJECT:               return strdup("Invalid memory object");
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return strdup("Invalid image format descriptor");
        case CL_INVALID_IMAGE_SIZE:               return strdup("Invalid image size");
        case CL_INVALID_SAMPLER:                  return strdup("Invalid sampler");
        case CL_INVALID_BINARY:                   return strdup("Invalid binary");
        case CL_INVALID_BUILD_OPTIONS:            return strdup("Invalid build options");
        case CL_INVALID_PROGRAM:                  return strdup("Invalid program");
        case CL_INVALID_PROGRAM_EXECUTABLE:       return strdup("Invalid program executable");
        case CL_INVALID_KERNEL_NAME:              return strdup("Invalid kernel name");
        case CL_INVALID_KERNEL_DEFINITION:        return strdup("Invalid kernel definition");
        case CL_INVALID_KERNEL:                   return strdup("Invalid kernel");
        case CL_INVALID_ARG_INDEX:                return strdup("Invalid argument index");
        case CL_INVALID_ARG_VALUE:                return strdup("Invalid argument value");
        case CL_INVALID_ARG_SIZE:                 return strdup("Invalid argument size");
        case CL_INVALID_KERNEL_ARGS:              return strdup("Invalid kernel arguments");
        case CL_INVALID_WORK_DIMENSION:           return strdup("Invalid work dimension");
        case CL_INVALID_WORK_GROUP_SIZE:          return strdup("Invalid work group size");
        case CL_INVALID_WORK_ITEM_SIZE:           return strdup("Invalid work item size");
        case CL_INVALID_GLOBAL_OFFSET:            return strdup("Invalid global offset");
        case CL_INVALID_EVENT_WAIT_LIST:          return strdup("Invalid event wait list");
        case CL_INVALID_EVENT:                    return strdup("Invalid event");
        case CL_INVALID_OPERATION:                return strdup("Invalid operation");
        case CL_INVALID_GL_OBJECT:                return strdup("Invalid OpenGL object");
        case CL_INVALID_BUFFER_SIZE:              return strdup("Invalid buffer size");
        case CL_INVALID_MIP_LEVEL:                return strdup("Invalid mip-map level");
        default:                                  return strdup("Unknown");
    }
}

#if defined(_WIN32) && !defined(_WIN64)
#define pfn_prefix __stdcall
#else
#define pfn_prefix
#endif

void pfn_prefix pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
	printf("\n\nOpenCL Error (via pfn_notify): %s\n\n", errinfo);
        fflush(stdout);
}

CryptohazeOpenCL::CryptohazeOpenCL() {
    
    this->platformsPopulated = 0;
    memset(this->OpenCLPlatforms, 0, MAX_SUPPORTED_PLATFORMS * sizeof(cl_platform_id));
    this->numPlatformsPopulated = 0;
    this->maxPlatformsSupported = MAX_SUPPORTED_PLATFORMS;
    this->currentPlatform = -1;

    this->devicesPopulated = 0;
    memset(this->OpenCLDevices, 0, MAX_SUPPORTED_DEVICES * sizeof(cl_device_id));
    this->numDevicesPopulated = 0;
    this->maxDevicesSupported = MAX_SUPPORTED_DEVICES;
    this->currentDevice = -1;

    this->contextPopulated = 0;

    this->programPopulated = 0;

    this->commandQueuePopulated = 0;

    this->defaultThreadCount = 64;
    this->defaultBlockCount = 64;
    
    this->dumpSourceFile = 0;
    
    this->deviceMask = CL_DEVICE_TYPE_ALL;
}

CryptohazeOpenCL::~CryptohazeOpenCL() {
    //printf("CryptohazeOpenCL::~CryptohazeOpenCL()\n");
    if (this->commandQueuePopulated) {
        clReleaseCommandQueue(this->OpenCLCommandQueue);
        //printf("Releasing command queue.\n");
    }
    if (this->contextPopulated) {
        //printf("Releasing context\n");
        clReleaseContext(this->OpenCLContext);
    }
}


void CryptohazeOpenCL::populatePlatforms() {
    cl_int errorCode;
    //printf("CryptohazeOpenCL::populatePlatforms()\n");
    fflush(stdout);
    // Get the available platforms.
    errorCode = clGetPlatformIDs(this->maxPlatformsSupported,
            this->OpenCLPlatforms, &this->numPlatformsPopulated);
    if (errorCode != CL_SUCCESS) {
        printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
        exit(1);
    }
    
    // Mark platforms as populated
    this->platformsPopulated = 1;

    //printf("Got %d platform(s)!\n", (int)this->numPlatformsPopulated);
}

int CryptohazeOpenCL::getNumberOfPlatforms() {
    if (!this->platformsPopulated) {
        this->populatePlatforms();
    }
    return this->numPlatformsPopulated;
}

void CryptohazeOpenCL::printAvailablePlatforms() {
    char platformInformationString[1024];
    int i;
    cl_int errorCode;

    if (!this->platformsPopulated) {
        this->populatePlatforms();
    }

    // Iterate through platforms
    for (i = 0; i < this->numPlatformsPopulated; i++) {
        printf("Platform ID %d: \n", (int)i);
        
        // I don't need how many bytes are filled - just don't overrun it, plz.
        // CL_PLATFORM_NAME
        errorCode = clGetPlatformInfo(this->OpenCLPlatforms[i], CL_PLATFORM_NAME, 1024, platformInformationString, NULL);
        if (errorCode != CL_SUCCESS) {
            printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
            exit(1);
        } else {
            printf("CL_PLATFORM_NAME: %s\n", platformInformationString);
        }

        // CL_PLATFORM_VENDOR
        errorCode = clGetPlatformInfo(this->OpenCLPlatforms[i], CL_PLATFORM_VENDOR, 1024, platformInformationString, NULL);
        if (errorCode != CL_SUCCESS) {
            printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
            exit(1);
        } else {
            printf("CL_PLATFORM_VENDOR: %s\n", platformInformationString);
        }

        // CL_PLATFORM_VERSION
        errorCode = clGetPlatformInfo(this->OpenCLPlatforms[i], CL_PLATFORM_VERSION, 1024, platformInformationString, NULL);
        if (errorCode != CL_SUCCESS) {
            printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
            exit(1);
        } else {
            printf("CL_PLATFORM_VERSION: %s\n", platformInformationString);
        }

        printf("\n\n");
    }
}

void CryptohazeOpenCL::selectPlatformById(int newPlatformId) {
    if (!this->platformsPopulated) {
        this->populatePlatforms();
    }
    this->currentPlatform = newPlatformId;
}

void CryptohazeOpenCL::populateDevices() {
    cl_int errorCode;

    if (this->currentPlatform == -1) {
        printf("Must select platform before querying devices!\n");
        exit(1);
    }

    // Get the available devices.
    errorCode = clGetDeviceIDs(this->OpenCLPlatforms[this->currentPlatform],
            this->deviceMask,
            this->maxDevicesSupported,
            this->OpenCLDevices,
            &this->numDevicesPopulated);
    
    if (errorCode != CL_SUCCESS) {
        printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
        exit(1);
    }
    // Mark platforms as populated
    this->devicesPopulated = 1;

    //printf("Got %d device(s)!\n", (int)this->numDevicesPopulated);
}

int CryptohazeOpenCL::getNumberOfDevices() {
    if (!this->devicesPopulated) {
        this->populateDevices();
    }
    return (int)this->numDevicesPopulated;
}

void CryptohazeOpenCL::printAvailableDevices() {
    char deviceInformationString[1024];
    cl_bool deviceBoolFlag;
    cl_uint deviceUintValue;
    cl_ulong deviceUlongValue;
    cl_device_local_mem_type deviceLocalMemType;
    cl_device_type deviceType;
    
    int i;
    cl_int errorCode;

    if (!this->devicesPopulated) {
        this->populateDevices();
    }

    // Iterate through platforms
    for (i = 0; i < this->numDevicesPopulated; i++) {
        printf("Device ID %d: \n", (int)i);

        // I don't need how many bytes are filled - just don't overrun it, plz.
        // CL_DEVICE_NAME
        errorCode = clGetDeviceInfo (this->OpenCLDevices[i],
                CL_DEVICE_NAME,
                1024, deviceInformationString, NULL);
        if (errorCode != CL_SUCCESS) {
            printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
            exit(1);
        } else {
            printf("CL_DEVICE_NAME: %s\n", deviceInformationString);
        }

        //CL_DEVICE_VENDOR
        errorCode = clGetDeviceInfo (this->OpenCLDevices[i],
                CL_DEVICE_VENDOR,
                1024, deviceInformationString, NULL);
        if (errorCode != CL_SUCCESS) {
            printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
            exit(1);
        } else {
            printf("CL_DEVICE_VENDOR: %s\n", deviceInformationString);
        }

        // CL_DEVICE_AVAILABLE
        errorCode = clGetDeviceInfo (this->OpenCLDevices[i],
                CL_DEVICE_AVAILABLE,
                sizeof(deviceBoolFlag), &deviceBoolFlag, NULL);
        if (errorCode != CL_SUCCESS) {
            printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
            exit(1);
        } else {
            printf("CL_DEVICE_AVAILABLE: %s\n", deviceBoolFlag ? "Yes" : "No");
        }

        // CL_DEVICE_MAX_COMPUTE_UNITS
        errorCode = clGetDeviceInfo (this->OpenCLDevices[i],
                CL_DEVICE_MAX_COMPUTE_UNITS,
                sizeof(deviceUintValue), &deviceUintValue, NULL);
        if (errorCode != CL_SUCCESS) {
            printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
            exit(1);
        } else {
            printf("CL_DEVICE_MAX_COMPUTE_UNITS: %d\n", deviceUintValue);
        }

        // CL_DEVICE_TYPE
        errorCode = clGetDeviceInfo (this->OpenCLDevices[i],
                CL_DEVICE_TYPE,
                sizeof(deviceType), &deviceType, NULL);
        if (errorCode != CL_SUCCESS) {
            printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
            exit(1);
        } else {
            switch(deviceType) {
                case CL_DEVICE_TYPE_CPU:
                    printf("CL_DEVICE_TYPE: CL_DEVICE_TYPE_CPU\n");
                    break;
                case CL_DEVICE_TYPE_GPU:
                    printf("CL_DEVICE_TYPE: CL_DEVICE_TYPE_GPU\n");
                    break;
                case CL_DEVICE_TYPE_ACCELERATOR:
                    printf("CL_DEVICE_TYPE: CL_DEVICE_TYPE_ACCELERATOR\n");
                    break;
                case CL_DEVICE_TYPE_DEFAULT:
                    printf("CL_DEVICE_TYPE: CL_DEVICE_TYPE_DEFAULT\n");
                    break;
                default:
                    printf("CL_DEVICE_TYPE: No idea... error?\n");
                    break;
            }
        }


        // CL_DEVICE_MAX_CLOCK_FREQUENCY
        errorCode = clGetDeviceInfo (this->OpenCLDevices[i],
                CL_DEVICE_MAX_CLOCK_FREQUENCY,
                sizeof(deviceUintValue), &deviceUintValue, NULL);
        if (errorCode != CL_SUCCESS) {
            printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
            exit(1);
        } else {
            printf("CL_DEVICE_MAX_CLOCK_FREQUENCY: %d MHz\n", deviceUintValue);
        }

        //CL_DEVICE_GLOBAL_MEM_SIZE
        errorCode = clGetDeviceInfo (this->OpenCLDevices[i],
                CL_DEVICE_GLOBAL_MEM_SIZE,
                sizeof(deviceUlongValue), &deviceUlongValue, NULL);
        if (errorCode != CL_SUCCESS) {
            printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
            exit(1);
        } else {
            printf("CL_DEVICE_GLOBAL_MEM_SIZE: %lu MB\n", (unsigned long)deviceUlongValue / (1024*1024));
        }

        // CL_DEVICE_LOCAL_MEM_SIZE
        errorCode = clGetDeviceInfo (this->OpenCLDevices[i],
                CL_DEVICE_LOCAL_MEM_SIZE,
                sizeof(deviceUlongValue), &deviceUlongValue, NULL);
        if (errorCode != CL_SUCCESS) {
            printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
            exit(1);
        } else {
            printf("CL_DEVICE_LOCAL_MEM_SIZE: %lu KB\n", (unsigned long)deviceUlongValue / 1024);
        }

        // CL_DEVICE_LOCAL_MEM_TYPE
        errorCode = clGetDeviceInfo (this->OpenCLDevices[i],
                CL_DEVICE_LOCAL_MEM_TYPE,
                sizeof(deviceLocalMemType), &deviceLocalMemType, NULL);
        if (errorCode != CL_SUCCESS) {
            printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
            exit(1);
        } else {
            printf("CL_DEVICE_LOCAL_MEM_TYPE: ");
            if (deviceLocalMemType == CL_LOCAL) {
                printf("CL_LOCAL\n");
            } else if (deviceLocalMemType == CL_GLOBAL) {
                printf("CL_GLOBAL\n");
            } else {
                printf("UNKNOWN\n");
            }
        }

        // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
        errorCode = clGetDeviceInfo (this->OpenCLDevices[i],
                CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                sizeof(deviceUlongValue), &deviceUlongValue, NULL);
        if (errorCode != CL_SUCCESS) {
            printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
            exit(1);
        } else {
            printf("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: %lu KB\n", (unsigned long)deviceUlongValue / 1024);
        }

        printf("\n\n");
    }
}

void CryptohazeOpenCL::selectDeviceById(int newDeviceId) {
    if (!this->devicesPopulated) {
        this->populateDevices();
    }
    this->currentDevice = newDeviceId;
    
    char platformInformationString[1024];
    cl_bool deviceBoolFlag;
    cl_uint deviceUintValue;
    cl_ulong deviceUlongValue;
    cl_device_local_mem_type deviceLocalMemType;
    cl_device_type deviceType;
    cl_int errorCode;

    // Figure out the good thread/block counts to use.
    //CL_PLATFORM_VENDOR
    errorCode = clGetPlatformInfo(this->OpenCLPlatforms[this->currentPlatform], 
            CL_PLATFORM_NAME, 1024, platformInformationString, NULL);
    if (errorCode != CL_SUCCESS) {
        printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
        exit(1);
    }
    
    // CL_DEVICE_TYPE
    errorCode = clGetDeviceInfo (this->OpenCLDevices[this->currentDevice],
            CL_DEVICE_TYPE,
            sizeof(deviceType), &deviceType, NULL);
    if (errorCode != CL_SUCCESS) {
        printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
        exit(1);
    }
    
    // Sort out what platform we have.
    
    // Apple is odd - their CPU type can only have one thread/block.
    if (strstr(platformInformationString, "Apple")) {
        if (deviceType == CL_DEVICE_TYPE_CPU) {
            this->defaultBlockCount = 1;
            this->defaultThreadCount = 1;
        }
    } else {
        if (deviceType == CL_DEVICE_TYPE_CPU) {
            this->defaultBlockCount = 16;
            this->defaultThreadCount = 1;
        } else if (deviceType == CL_DEVICE_TYPE_GPU) {
            this->defaultBlockCount = 256;
            this->defaultThreadCount = 256;
        }
    }
    
}

void CryptohazeOpenCL::createContext() {
    cl_int errorCode;

    // Create the context specifying the platform
    cl_context_properties contextProperties[] =
        {
        CL_CONTEXT_PLATFORM, (cl_context_properties) this->OpenCLPlatforms[this->currentPlatform], 0
        };

    if (this->currentPlatform == -1) {
        printf("Must select a platform before creating a context!\n");
        exit(1);
    }
    if (this->currentDevice == -1) {
        printf("Must select a device before creating a context!\n");
        exit(1);
    }

    //void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data)

    this->OpenCLContext =   clCreateContext (contextProperties,
            1 /* num_devices */,
            &this->OpenCLDevices[this->currentDevice],
            pfn_notify,
            NULL /* user data */,
            &errorCode);
    if (errorCode != CL_SUCCESS) {
        printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
        exit(1);
    } else {
        //printf("Context Created!\n");
        this->contextPopulated = 1;
    }
}

cl_context CryptohazeOpenCL::getContext() {
    return this->OpenCLContext;
}


void CryptohazeOpenCL::buildProgramFromSource(const char *filename, const char *options) {
    cl_int errorCode;
    char *programSource;
    size_t programSourceSize;

    if (!this->contextPopulated) {
        printf("Error: Cannot call buildProgramFromSource with no context!\n");
        exit(1);
    }
    if (!this->devicesPopulated) {
        printf("Error: Cannot call buildProgramFromSource with no devices!\n");
        exit(1);
    }

    // Read from the file into a string.
    fstream sourceFile(filename, (std::fstream::in | std::fstream::binary));

    if(sourceFile.is_open()) {
        size_t fileSize;
        sourceFile.seekg(0, std::fstream::end);
        programSourceSize = fileSize = (size_t)sourceFile.tellg();
        sourceFile.seekg(0, std::fstream::beg);

        programSource = new char[programSourceSize+1];
        if(!programSource)  {
            printf("Alloc failed for programSource!\n");
            exit(1);
        }
        memset(programSource, 0, programSourceSize+1);

        sourceFile.read(programSource, fileSize);
        sourceFile.close();
    } else {
        printf("Cannot open source file %s!\n", filename);
        exit(1);
    }

    // Theoretically, source loaded.  Let's try to compile it!
    this->OpenCLProgram = clCreateProgramWithSource (this->OpenCLContext,
            1 /* Number of strings */,
            (const char **)&programSource,
            NULL /* Null terminated strings used */,
            &errorCode);

    if (errorCode != CL_SUCCESS) {
        printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
        exit(1);
    } else {
        //printf("clCreateProgramWithSource succeeded!\n");
    }

    //printf("Attempting to compile...\n");
    // Now, try to actually build it.
    errorCode =  clBuildProgram (this->OpenCLProgram,
            1 /* num_devices */,
            &this->OpenCLDevices[this->currentDevice],
            options,
            NULL,
            NULL);

    //printf("Compilation complete.\n");

    if (errorCode != CL_SUCCESS) {
        printf("Error compiling %s!\n", filename);
        printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
        char buildErrorString[1000000];
        memset(buildErrorString, 0, sizeof(buildErrorString));
        errorCode = clGetProgramBuildInfo(this->OpenCLProgram,
                this->OpenCLDevices[this->currentDevice],
                CL_PROGRAM_BUILD_LOG,
                sizeof(char) * sizeof(buildErrorString), buildErrorString, NULL);
        if(errorCode != CL_SUCCESS) {
            printf("Could not get build log: %s\n", print_cl_errstring(errorCode));
            exit(1);
        }
        printf("Build log: \n%s\n", buildErrorString);
        exit(1);
    }
    
    this->programPopulated = 1;
    
    delete[] programSource;
}


void CryptohazeOpenCL::buildProgramFromManySources(std::vector<std::string> sourceFilenames, const char *options) {
    cl_int errorCode;
    std::vector<std::string> programSources;
    const char **programSourcesToBuild;
    int fileCount;


    if (!this->contextPopulated) {
        printf("Error: Cannot call buildProgramFromSource with no context!\n");
        exit(1);
    }
    if (!this->devicesPopulated) {
        printf("Error: Cannot call buildProgramFromSource with no devices!\n");
        exit(1);
    }

    for (fileCount = 0; fileCount < sourceFilenames.size(); fileCount++) {
        fstream sourceFile(sourceFilenames.at(fileCount).c_str(), (std::fstream::in | std::fstream::binary));
        if(sourceFile.is_open()) {
            size_t fileSize;
            std::string programSource;
            char *programSourceCString;

            sourceFile.seekg(0, std::fstream::end);
            fileSize = (size_t)sourceFile.tellg();
            sourceFile.seekg(0, std::fstream::beg);

            programSourceCString = new char[fileSize+1];

            sourceFile.read(programSourceCString, fileSize);
            sourceFile.close();

            programSource = programSourceCString;
            programSources.push_back(programSource);
            delete[] programSourceCString;
            
        } else {
            printf("Cannot open source file %s!\n", sourceFilenames.at(fileCount).c_str());
            exit(1);
        }

    }

    programSourcesToBuild = new const char*[programSources.size()];

    for (fileCount = 0; fileCount < programSources.size(); fileCount++) {
        programSourcesToBuild[fileCount] = programSources[fileCount].c_str();
    }

    printf("Number of files: %d\n", programSources.size());

    // Theoretically, source loaded.  Let's try to compile it!
    this->OpenCLProgram = clCreateProgramWithSource (this->OpenCLContext,
            programSources.size() /* Number of strings */,
            programSourcesToBuild,
            NULL /* Null terminated strings used */,
            &errorCode);

    if (errorCode != CL_SUCCESS) {
        printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
        exit(1);
    } else {
        //printf("clCreateProgramWithSource succeeded!\n");
    }

    //printf("Attempting to compile...\n");
    // Now, try to actually build it.
    errorCode =  clBuildProgram (this->OpenCLProgram,
            1 /* num_devices */,
            &this->OpenCLDevices[this->currentDevice],
            options,
            NULL,
            NULL);

    //printf("Compilation complete.\n");

    if (errorCode != CL_SUCCESS) {
        printf("Error compiling program!\n");
        printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
        char buildErrorString[1000000];
        memset(buildErrorString, 0, sizeof(buildErrorString));
        errorCode = clGetProgramBuildInfo(this->OpenCLProgram,
                this->OpenCLDevices[this->currentDevice],
                CL_PROGRAM_BUILD_LOG,
                sizeof(char) * sizeof(buildErrorString), buildErrorString, NULL);
        if(errorCode != CL_SUCCESS) {
            printf("Could not get build log: %s\n", print_cl_errstring(errorCode));
            exit(1);
        }
        printf("Build log: \n%s\n", buildErrorString);
        exit(1);
    }

    this->programPopulated = 1;

    for (fileCount = 0; fileCount < programSources.size(); fileCount++) {
        //delete[] programSourcesToBuild[fileCount];
    }

    //delete[] programSourcesToBuild;
}

void CryptohazeOpenCL::buildProgramFromManySourcesConcat(std::vector<std::string> sourceFilenames, 
        const char *options, std::string prependDefines) {
    cl_int errorCode;
    std::string programSourceCode;
    int fileCount;
    const char *programSourceCstr;


    if (!this->contextPopulated) {
        printf("Error: Cannot call buildProgramFromSource with no context!\n");
        exit(1);
    }
    if (!this->devicesPopulated) {
        printf("Error: Cannot call buildProgramFromSource with no devices!\n");
        exit(1);
    }
    
    // Add the prepend code if present.
    programSourceCode += prependDefines;
    
    for (fileCount = 0; fileCount < sourceFilenames.size(); fileCount++) {
        fstream sourceFile(sourceFilenames.at(fileCount).c_str(), (std::fstream::in | std::fstream::binary));
        if(sourceFile.is_open()) {
            size_t fileSize;
            std::string programSource;
            char *programSourceCString;

            sourceFile.seekg(0, std::fstream::end);
            fileSize = (size_t)sourceFile.tellg();
            sourceFile.seekg(0, std::fstream::beg);

            programSourceCString = new char[fileSize+1];
            memset(programSourceCString, 0, fileSize+1);

            sourceFile.read(programSourceCString, fileSize);
            sourceFile.close();

            programSource = programSourceCString;
            programSourceCode += programSource;
            delete[] programSourceCString;
            
        } else {
            printf("Cannot open source file %s!\n", sourceFilenames.at(fileCount).c_str());
            exit(1);
        }

    }
    
    programSourceCstr = programSourceCode.c_str();

    if (this->dumpSourceFile) {
        FILE *outputDump;
        outputDump = fopen("/tmp/src.out", "w");
        fwrite(programSourceCstr,  programSourceCode.size(), 1, outputDump);
        fclose(outputDump);
    }

    // Theoretically, source loaded.  Let's try to compile it!
    this->OpenCLProgram = clCreateProgramWithSource (this->OpenCLContext,
            1 /* Number of strings */,
            (const char **)&programSourceCstr,
            NULL /* Null terminated strings used */,
            &errorCode);

    if (errorCode != CL_SUCCESS) {
        printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
        exit(1);
    } else {
        //printf("clCreateProgramWithSource succeeded!\n");
    }

    //printf("Attempting to compile...\n");
    // Now, try to actually build it.
    errorCode =  clBuildProgram (this->OpenCLProgram,
            1 /* num_devices */,
            &this->OpenCLDevices[this->currentDevice],
            options,
            NULL,
            NULL);

    //printf("Compilation complete.\n");

    if (errorCode != CL_SUCCESS) {
        printf("Error compiling program!\n");
        printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
        char buildErrorString[1000000];
        memset(buildErrorString, 0, sizeof(buildErrorString));
        errorCode = clGetProgramBuildInfo(this->OpenCLProgram,
                this->OpenCLDevices[this->currentDevice],
                CL_PROGRAM_BUILD_LOG,
                sizeof(char) * sizeof(buildErrorString), buildErrorString, NULL);
        if(errorCode != CL_SUCCESS) {
            printf("Could not get build log: %s\n", print_cl_errstring(errorCode));
            exit(1);
        }
        printf("Build log: \n%s\n", buildErrorString);
        exit(1);
    }

    this->programPopulated = 1;

    //delete[] programSourcesToBuild;
}

cl_program CryptohazeOpenCL::getProgram() {
    return this->OpenCLProgram;
}

// Pulls from gat3way's hashkill to do a BFI_INT patch.
void CryptohazeOpenCL::doAMDBFIPatch() {
    size_t nDevices=1;
    size_t * binary_sizes;
    unsigned char ** binaries;
    cl_int errorCode;
    int i;
    
    BFIPatcher patcher;

    if (!this->programPopulated) {
        printf("Cannot do BFI_INT patch before program is built!\n");
        exit(1);
    }

    errorCode = clGetProgramInfo(this->OpenCLProgram, CL_PROGRAM_NUM_DEVICES, sizeof(nDevices), &nDevices, NULL );
    binary_sizes = (size_t *)malloc( sizeof(size_t)*nDevices );
    errorCode = clGetProgramInfo(this->OpenCLProgram, CL_PROGRAM_BINARY_SIZES, sizeof(size_t)*nDevices, binary_sizes, NULL );

    //printf("Number of devices: %d\n", nDevices);

    binaries = (unsigned char **)malloc( sizeof(unsigned char *)*nDevices );
    for( i = 0; i < nDevices; i++ )
    {
        if( binary_sizes[i] != 0 )
        {
            binaries[i] = (unsigned char *)malloc( sizeof(unsigned char)*binary_sizes[i]+100 );
            memset(binaries[i], 0, binary_sizes[i]+100);
        }
        else
        {
            binaries[i] = NULL;
        }
    }
    clGetProgramInfo(this->OpenCLProgram, CL_PROGRAM_BINARIES, sizeof(char *)*nDevices, binaries, NULL );
    if (0)
    {
        printf("\n");
        printf("Binary size: %d\n",binary_sizes[0]);
        printf("Doing BFI_INT magic...%s\n","");
    }
    if (0) {
        FILE *binaryoutput;

        binaryoutput = fopen("amd.bin", "wb");
        // DERP.  Write from the array starting at binaries[0], not the address
        // of binaries[0] derp derp derp
        fwrite(binaries[0], binary_sizes[0], 1, binaryoutput);
        fclose(binaryoutput);
    }
    
    std::vector<uint8_t> amdBinary;
    amdBinary.resize(binary_sizes[0], 0);
    
    memcpy(&amdBinary[0], binaries[0], binary_sizes[0]);
    amdBinary = patcher.patchBinary(amdBinary);
    memcpy(binaries[0], &amdBinary[0], binary_sizes[0]);

    this->OpenCLProgram =
            clCreateProgramWithBinary(this->OpenCLContext, 1,
            &this->OpenCLDevices[this->currentDevice], &binary_sizes[0],
            (const unsigned char **)&binaries[0], NULL, &errorCode );
    if (errorCode!=CL_SUCCESS) {
        printf("Cannot compile binary!\n%s","");
        exit(1);
    }
    errorCode = clBuildProgram(this->OpenCLProgram, 1, &this->OpenCLDevices[this->currentDevice], NULL, NULL, NULL );
    if (errorCode!=CL_SUCCESS) {
        printf("Cannot build binary!\n%s","");
        exit(1);
    }
}


cl_ulong CryptohazeOpenCL::getMaximumAllocSize() {
    cl_ulong deviceUlongValue;
    cl_int errorCode;

    if (!this->devicesPopulated) {
        this->populateDevices();
    }

    errorCode = clGetDeviceInfo (this->OpenCLDevices[this->currentDevice],
            CL_DEVICE_MAX_MEM_ALLOC_SIZE,
            sizeof(deviceUlongValue), &deviceUlongValue, NULL);
    if (errorCode != CL_SUCCESS) {
        printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
        exit(1);
    } else {
        return deviceUlongValue;
    }
}

cl_ulong CryptohazeOpenCL::getGlobalMemorySize() {
    cl_ulong deviceUlongValue;
    cl_int errorCode;

    if (!this->devicesPopulated) {
        this->populateDevices();
    }

    errorCode = clGetDeviceInfo (this->OpenCLDevices[this->currentDevice],
            CL_DEVICE_GLOBAL_MEM_SIZE,
            sizeof(deviceUlongValue), &deviceUlongValue, NULL);
    if (errorCode != CL_SUCCESS) {
        printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
        exit(1);
    } else {
        return deviceUlongValue;
    }
}

cl_ulong CryptohazeOpenCL::getLocalMemorySize() {
    cl_ulong deviceUlongValue;
    cl_int errorCode;

    if (!this->devicesPopulated) {
        this->populateDevices();
    }

    errorCode = clGetDeviceInfo (this->OpenCLDevices[this->currentDevice],
            CL_DEVICE_LOCAL_MEM_SIZE,
            sizeof(deviceUlongValue), &deviceUlongValue, NULL);
    if (errorCode != CL_SUCCESS) {
        printf("Error: (%s:%d)%s\n",__FILE__, __LINE__, print_cl_errstring(errorCode));
        exit(1);
    } else {
        return deviceUlongValue;
    }
}

void CryptohazeOpenCL::createCommandQueue() {
    cl_int errorCode;

    this->OpenCLCommandQueue = clCreateCommandQueue (this->OpenCLContext,
            this->OpenCLDevices[this->currentDevice],
            NULL,
            &errorCode);
    if (errorCode != CL_SUCCESS) {
        printf("Error: %s\n", print_cl_errstring(errorCode));
        exit(1);
    }
    this->commandQueuePopulated = 1;
}

cl_command_queue CryptohazeOpenCL::getCommandQueue() {
    return this->OpenCLCommandQueue;
}

int CryptohazeOpenCL::getDefaultThreadCount() {
    return this->defaultThreadCount;
}
int CryptohazeOpenCL::getDefaultBlockCount() {
    return this->defaultBlockCount;
}

void CryptohazeOpenCL::setDeviceType(uint32_t newDeviceType) {
    switch(newDeviceType) {
        case DEVICE_CPU:
            this->deviceMask = CL_DEVICE_TYPE_CPU;
            break;
        case DEVICE_GPU:
            this->deviceMask = CL_DEVICE_TYPE_GPU;
            break;
        case DEVICE_ALL:
            this->deviceMask = CL_DEVICE_TYPE_ALL;
            break;
        default:
            this->deviceMask = CL_DEVICE_TYPE_ALL;
            break;
    }
}

#if UNIT_TEST
int main() {
    CryptohazeOpenCL *OpenCL;
    cl_context myContext;

    OpenCL = new CryptohazeOpenCL();

    OpenCL->printAvailablePlatforms();
    OpenCL->selectPlatformById(0);

    OpenCL->printAvailableDevices();
    OpenCL->selectDeviceById(0);

    OpenCL->createContext();

    OpenCL->buildProgramFromSource("GRTCLTest.cl");
    
}
#endif

