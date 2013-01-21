CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )
PROJECT ( cryptohaze )

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET ( CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/bin 
          CACHE PATH "Directory to install binaries to" FORCE 
        )
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)


# Check if we're building 32bit 
IF ( CMAKE_SIZEOF_VOID_P EQUAL 4 )
    SET ( CH_BUILD_32BIT ON CACHE BOOL "Build cryptohaze as 32bit target using 32bit libs" )
ENDIF ( CMAKE_SIZEOF_VOID_P EQUAL 4 )

IF ( WIN32 ) 
    SET ( CH_LIB_FOLDER windows_x64 )
ELSE ()
    SET ( CH_LIB_FOLDER linux_64 )
    SET ( CH_THREAD_LIBRARIES pthread )
ENDIF ( WIN32 )

OPTION ( CH_LINK_AGAINST_PLATFORM_DIR 
         "Add the libs/${CH_LIB_FOLDER} to the library search path" OFF 
       )
IF ( CH_LINK_AGAINST_PLATFORM_DIR )
    SET ( CH_PLATFORM_LINK_DIR ${cryptohaze_SOURCE_DIR}/libs/${CH_LIB_FOLDER} 
            CACHE PATH "Additional folder that dependency libs can be placed in" )
ENDIF ( CH_LINK_AGAINST_PLATFORM_DIR )

# Add in all the gencode options
OPTION ( CH_GENCODE_10 "Add gencode compute_10" ON )
IF ( CH_GENCODE_10 )
    SET ( GENCODE_SM_10_FLAGS "-gencode=arch=compute_10,code=sm_10;" 
                              "-gencode=arch=compute_10,code=compute_10;"
        ) 
ENDIF ( CH_GENCODE_10 )

OPTION ( CH_GENCODE_20 "Add gencode compute_20" ON )
IF ( CH_GENCODE_20 )
    SET ( GENCODE_SM_20_FLAGS -gencode=arch=compute_20,code=sm_20;
                              -gencode=arch=compute_20,code=compute_20;
        ) 
ENDIF ( CH_GENCODE_20 )

OPTION ( CH_GENCODE_30 "Add gencode compute_30" ON )
IF ( CH_GENCODE_30 )
    SET ( GENCODE_SM_30_FLAGS -gencode=arch=compute_30,code=sm_30; 
                              -gencode=arch=compute_30,code=compute_30;
        ) 
ENDIF ( CH_GENCODE_30 )

# Option to allow user to add additional gencode archs
SET ( CH_GENCODE_ADDITIONAL_ARCH "" 
      CACHE STRING "Allows specification of additional NVCC gencode flags" 
    )

# Option to enable debugging symbols in the binaries
OPTION ( CH_DEBUGGING_SYMBOLS "Add debugging symbols in the binary" OFF )
IF ( CH_DEBUGGING_SYMBOLS )
    SET(CMAKE_CXX_FLAGS "-g -Wall")
ENDIF (CH_DEBUGGING_SYMBOLS )

OPTION ( CH_STATIC_LINKING "Statically link as many libraries as possible" OFF )
IF ( CH_STATIC_LINKING ) 
    IF ( WIN32 ) 
    ELSE()
        SET (CMAKE_FIND_LIBRARY_SUFFIXES .a .so)
        ADD_DEFINITIONS ( -DCURL_STATICLIB ) 
    ENDIF ( WIN32 )
ENDIF ( CH_STATIC_LINKING )
# Locate required libraries
#--------------------------

# Find CUDA libs/includes
FIND_PACKAGE ( CUDA )
IF ( NOT CUDA_FOUND)
    MESSAGE ( FATAL_ERROR "ERROR:  CUDA required, not found" )
ENDIF ( NOT CUDA_FOUND )

SET ( CUDA_NVCC_FLAGS
      "${CUDA_NVCC_FLAGS}"
      "${GENCODE_SM_10_FLAGS}"
      "${GENCODE_SM_20_FLAGS}"
      "${GENCODE_SM_30_FLAGS}"
      "${CH_GENCODE_ADDITIONAL_ARCH}"
      "--compiler-options;-fno-strict-aliasing"
      )

#MESSAGE ( "SET NVCC FLAGS TO "  ${CUDA_NVCC_FLAGS})

# Find/set OpenCL/CUDA SDK headers + lib 
IF ( WIN32 )
    SET ( OPENCL_INCLUDE_DIR ${CUDA_SDK_ROOT_DIR}/../OpenCL/common/inc )
    SET ( OPENCL_LIBRARIES OpenCL.lib )

    IF ( CH_BUILD_32BIT )
        SET ( OPENCL_LIBRARIES_DIR ${CUDA_SDK_ROOT_DIR}/../OpenCL/common/lib/Win32/ )
        SET ( CUDA_SDK_LIBRARIES_DIR ${CUDA_SDK_ROOT_DIR}/common/lib/Win32/ )
        # SET ( CUDA_SDK_LIBRARIES cutil32.lib; )
    ELSE ( )
        SET ( OPENCL_LIBRARIES_DIR ${CUDA_SDK_ROOT_DIR}/../OpenCL/common/lib/x64/ )
        SET ( CUDA_SDK_LIBRARIES_DIR ${CUDA_SDK_ROOT_DIR}/common/lib/x64/ )
        # SET ( CUDA_SDK_LIBRARIES cutil64.lib; )
    ENDIF ( CH_BUILD_32BIT )

    #TODO copy cutil64/32.dll to bin/

ELSE ()
    SET ( OPENCL_INCLUDE_DIR ${CUDA_SDK_ROOT_DIR}/../OpenCL/common/inc )

    # Apple requires the "framework OpenCL" flags instead of the library.
    IF ( APPLE )
        FIND_LIBRARY( OPENCL_LIBRARIES OpenCL )
    ELSE ()
        SET ( OPENCL_LIBRARIES OpenCL )
    ENDIF ( APPLE )
ENDIF ( WIN32 )


# Find boost headers/libs
SET ( BOOST_MIN_VERSION "1.48.0" )
# Force static boost libs and have it show up in CMake config 
SET ( Boost_USE_STATIC_LIBS ON CACHE BOOL "Compile against static boost" )

IF ( WIN32 )
    # TODO - how to avoid using FORCE on these - nulls any user changes but 
    # without it, switching from 32 to 64 and back won't update the variable 
    # correctly Also, FIND_PACKAGE after switching BOOST_LIBRARYDIR won't search 
    # the new dir for the libs and keeps the old lib dir cached... even if the 
    # user wants to try to change this, they can't :(
    IF ( CH_BUILD_32BIT )
        SET ( BOOST_LIBRARYDIR 
              $ENV{BOOST_ROOT}/stage/lib/x86 
              CACHE PATH "Boost lib folder" FORCE 
            )
    ELSE () 
        SET ( BOOST_LIBRARYDIR 
              $ENV{BOOST_ROOT}/stage/lib/x64 
              CACHE PATH "Boost lib folder" FORCE 
            )
    ENDIF ( CH_BUILD_32BIT )
ENDIF ( WIN32 )

IF ( WIN32 )
    FIND_PACKAGE ( Boost ${BOOST_MIN_VERSION} 
        COMPONENTS random system filesystem thread date_time regex iostreams 
        )
ELSE ()
    FIND_PACKAGE ( Boost ${BOOST_MIN_VERSION} 
        COMPONENTS random system filesystem thread 
        )
ENDIF ( WIN32 )
IF ( NOT Boost_FOUND )
    MESSAGE ( FATAL_ERROR "Minimum version of Boost ${BOOST_MIN_VERSION} required" )
ENDIF ( NOT Boost_FOUND )

# Find curl
IF ( WIN32 ) 
    SET ( CURL_LIBRARIES libcurl.lib;wldap32.lib;ws2_32.lib )
    SET ( CURL_INCLUDE_DIRS ${cryptohaze_SOURCE_DIR}/dependencies/curl-7.26.0/include 
            CACHE PATH "curl includes directory" )

    # TODO - any way to validate libcurl.lib exists here?
    SET ( CURL_LIBRARY_DIR ${cryptohaze_SOURCE_DIR}/dependencies/curl-7.26.0/lib
            CACHE PATH "curl library directory" )
ELSE ()
    IF ( CH_STATIC_LINKING )
        # Static libcurl requires librt.a
        SET ( CURL_LIBRARIES libcurl.a librt.a; )
        SET ( CURL_INCLUDE_DIRS ${cryptohaze_SOURCE_DIR}/dependencies/curl-7.26.0/include 
                CACHE PATH "curl includes directory" )

        # TODO - any way to validate libcurl.lib exists here?
        SET ( CURL_LIBRARY_DIR ${cryptohaze_SOURCE_DIR}/dependencies/curl-7.26.0/lib
                CACHE PATH "curl library directory" )
    ELSE ()
        FIND_PACKAGE(CURL REQUIRED)
    ENDIF ( CH_STATIC_LINKING )
ENDIF ( WIN32 )

# Find ncurses
IF ( WIN32 ) 
    SET( CURSES_INCLUDE_DIR ${cryptohaze_SOURCE_DIR}/dependencies/pdcurs34 
        CACHE PATH "pdcurs34 includes directory" )

    # gives us a "pdcurs34" lib
    ADD_SUBDIRECTORY ( ${cryptohaze_SOURCE_DIR}/dependencies/pdcurs34 )
    SET ( CURSES_LIBRARIES pdcurs34; )

ELSE ()
    IF ( CH_STATIC_LINKING )
        SET (CMAKE_FIND_LIBRARY_SUFFIXES .so)
    ENDIF ( CH_STATIC_LINKING )
    
    SET ( CURSES_NEED_NCURSES TRUE )
    
    FIND_PACKAGE(Curses REQUIRED)
    IF ( CH_STATIC_LINKING )
        SET (CMAKE_FIND_LIBRARY_SUFFIXES .a .so)
    ENDIF ( CH_STATIC_LINKING )
ENDIF ( WIN32 ) 

# On windows, setup argtable.  Allows for include <argtable2.h> and generates a 
# linkable target, argtable2
IF ( WIN32 )
    ADD_SUBDIRECTORY ( ${cryptohaze_SOURCE_DIR}/dependencies/argtable2-13 )
    SET ( ARGTABLE_INCLUDE_DIR ${cryptohaze_SOURCE_DIR}/dependencies/argtable2-13/src 
        CACHE PATH "argtable includes directory" )
ENDIF ( WIN32 )

IF ( WIN32 )
    # Provide hint for finding protobuf source.  FindProtobuf will look in the
    # vsprojects folder for debug/release libs
    SET ( PROTOBUF_SRC_ROOT_FOLDER ${cryptohaze_SOURCE_DIR}/dependencies/protobuf-2.4.1 
        CACHE PATH "Source root folder for protobuf" )
ENDIF ( WIN32 )

FIND_PACKAGE( Protobuf REQUIRED ) 

# Setup include paths and library paths
#--------------------------------------

# Main include dir all cryptohaze sources rely on
SET ( CH_INCLUDE_DIR ${cryptohaze_SOURCE_DIR}/inc )

# All include directories combined (boost, curses, curl, cuda, etc)
SET ( INCLUDE_DIRS 
    ${CH_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${CUDA_SDK_ROOT_DIR}/common/inc
    ${OPENCL_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    ${CURSES_INCLUDE_DIR}
    ${ARGTABLE_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIRS}
    )

INCLUDE_DIRECTORIES ( ${INCLUDE_DIRS} )
CUDA_INCLUDE_DIRECTORIES ( ${INCLUDE_DIRS} )

#SET ( CUDA_NVCC_FLAGS 
#      ${CUDA_NVCC_FLAGS}
#      ${GENCODE_SM_10_FLAGS}
#      ${GENCODE_SM_20_FLAGS}
#      ${GENCODE_SM_30_FLAGS}
#      ${CH_GENCODE_ADDITIONAL_ARCH}
#      --compiler-options -fno-strict-aliasing
#      CACHE STRING "NVCC compilier flags"
#      )

IF ( WIN32 )
    # strips down windows.h to resolve some errors including winsock 
    ADD_DEFINITIONS ( -DWIN32_LEAN_AND_MEAN ) 
ELSE ( WIN32 )
    # Add pipeline build for g++
    SET(CMAKE_CXX_FLAGS "-pipe")
ENDIF ( WIN32 )


LINK_DIRECTORIES (
    ${CUDA_SDK_LIBRARIES_DIR}
    ${CH_PLATFORM_LINK_DIR}
    ${Boost_LIBRARY_DIRS}
    ${OPENCL_LIBRARIES_DIR}
    ${CH_PLATFORM_LINK_DIR}
    ${CURL_LIBRARY_DIR} 
    )

ADD_SUBDIRECTORY ( shared/ )
ADD_SUBDIRECTORY ( src/ )
