PROJECT ( grt_opencl_device )

# Defines macro for calling MakeHeaderFromFile.  SOURCE_NAME should be the name
# of the source file - ex input FileX.cl should just pass FileX with no 
# extension.  It will create a FileX_source.h in the same folder
MACRO ( MAKE_GRTOPENCL_HEADER_FROM_FILE SOURCE_NAME )
    ADD_CUSTOM_COMMAND (
        OUTPUT ${CH_INCLUDE_DIR}/GRT_OpenCL_device/${SOURCE_NAME}_source.h
        COMMAND MakeHeaderFromFile
            ${grt_opencl_device_SOURCE_DIR}/${SOURCE_NAME}.cl
            ${SOURCE_NAME}_source
            ${CH_INCLUDE_DIR}/GRT_OpenCL_device/${SOURCE_NAME}_source.h
        DEPENDS MakeHeaderFromFile 
                ${grt_opencl_device_SOURCE_DIR}/${SOURCE_NAME}.cl
        )

    SET_SOURCE_FILES_PROPERTIES(
        ${grt_opencl_device_SOURCE_DIR}/${SOURCE_NAME}_source.h
        PROPERTIES GENERATED TRUE
        )
ENDMACRO ( MAKE_GRTOPENCL_HEADER_FROM_FILE SOURCE_NAME )

MAKE_GRTOPENCL_HEADER_FROM_FILE(GRT_OpenCL_Common)

MAKE_GRTOPENCL_HEADER_FROM_FILE(GRT_OpenCL_MD5)
MAKE_GRTOPENCL_HEADER_FROM_FILE(GRT_OpenCL_NTLM)
MAKE_GRTOPENCL_HEADER_FROM_FILE(GRT_OpenCL_SHA1)
MAKE_GRTOPENCL_HEADER_FROM_FILE(GRT_OpenCL_SHA256)

MAKE_GRTOPENCL_HEADER_FROM_FILE(GRTCLGenerateTableMD5_AMD)
MAKE_GRTOPENCL_HEADER_FROM_FILE(GRTCLGenerateTableNTLM_AMD)
MAKE_GRTOPENCL_HEADER_FROM_FILE(GRTCLGenerateTableSHA1_AMD)
MAKE_GRTOPENCL_HEADER_FROM_FILE(GRTCLGenerateTableSHA256_AMD)

MAKE_GRTOPENCL_HEADER_FROM_FILE(GRTCLCandidateHashesMD5)
MAKE_GRTOPENCL_HEADER_FROM_FILE(GRTCLCandidateHashesNTLM)
MAKE_GRTOPENCL_HEADER_FROM_FILE(GRTCLCandidateHashesSHA1)
MAKE_GRTOPENCL_HEADER_FROM_FILE(GRTCLCandidateHashesSHA256)

MAKE_GRTOPENCL_HEADER_FROM_FILE(GRTCLRegenerateChainsMD5)
MAKE_GRTOPENCL_HEADER_FROM_FILE(GRTCLRegenerateChainsNTLM)
MAKE_GRTOPENCL_HEADER_FROM_FILE(GRTCLRegenerateChainsSHA1)
MAKE_GRTOPENCL_HEADER_FROM_FILE(GRTCLRegenerateChainsSHA256)

# the generated TRUE isn't carrying over to other projects so we just force the 
# headers to always get created with ALL DEPENDS
ADD_CUSTOM_TARGET ( grt_opencl_device_headers echo "Creating GRT OpenCL Device Source Headers"
    DEPENDS
    ${CH_INCLUDE_DIR}/GRT_OpenCL_device/GRT_OpenCL_Common_source.h

    ${CH_INCLUDE_DIR}/GRT_OpenCL_device/GRT_OpenCL_MD5_source.h
    ${CH_INCLUDE_DIR}/GRT_OpenCL_device/GRT_OpenCL_NTLM_source.h
    ${CH_INCLUDE_DIR}/GRT_OpenCL_device/GRT_OpenCL_SHA1_source.h
    ${CH_INCLUDE_DIR}/GRT_OpenCL_device/GRT_OpenCL_SHA256_source.h

    ${CH_INCLUDE_DIR}/GRT_OpenCL_device/GRTCLGenerateTableMD5_AMD_source.h
    ${CH_INCLUDE_DIR}/GRT_OpenCL_device/GRTCLGenerateTableNTLM_AMD_source.h
    ${CH_INCLUDE_DIR}/GRT_OpenCL_device/GRTCLGenerateTableSHA1_AMD_source.h
    ${CH_INCLUDE_DIR}/GRT_OpenCL_device/GRTCLGenerateTableSHA256_AMD_source.h

    ${CH_INCLUDE_DIR}/GRT_OpenCL_device/GRTCLCandidateHashesMD5_source.h
    ${CH_INCLUDE_DIR}/GRT_OpenCL_device/GRTCLCandidateHashesNTLM_source.h
    ${CH_INCLUDE_DIR}/GRT_OpenCL_device/GRTCLCandidateHashesSHA1_source.h
    ${CH_INCLUDE_DIR}/GRT_OpenCL_device/GRTCLCandidateHashesSHA256_source.h

    ${CH_INCLUDE_DIR}/GRT_OpenCL_device/GRTCLRegenerateChainsMD5_source.h
    ${CH_INCLUDE_DIR}/GRT_OpenCL_device/GRTCLRegenerateChainsNTLM_source.h
    ${CH_INCLUDE_DIR}/GRT_OpenCL_device/GRTCLRegenerateChainsSHA1_source.h
    ${CH_INCLUDE_DIR}/GRT_OpenCL_device/GRTCLRegenerateChainsSHA256_source.h
    )

ADD_DEPENDENCIES ( grt_opencl_device_headers MakeHeaderFromFile )


FILE ( GLOB GRT_OPENCL_DEVICE_KERNELS *.cl )
FILE ( GLOB GRT_OPENCL_DEVICE_KERNELS_HEADERS ${CH_INCLUDE_DIR}/GRT_OpenCL_device/*.h )


# Shouldn't need the kernels installed to /kernels except for devel.
#INSTALL ( 
#    FILES ${GRT_OPENCL_DEVICE_KERNELS} 
#    DESTINATION ${CMAKE_INSTALL_PREFIX}/kernels 
#    )
#INSTALL ( 
#    FILES ${GRT_OPENCL_DEVICE_KERNELS_HEADERS} 
#    DESTINATION ${CMAKE_INSTALL_PREFIX}/kernels 
#    )
