PROJECT ( grt_opencl_host )

SET ( GRT_OPENCL_HOST_SOURCES
    ${grt_opencl_host_SOURCE_DIR}/GRTCLCandidateHashes.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLCandidateHashesMD5.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLCandidateHashesNTLM.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLCandidateHashesSHA1.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLCandidateHashesSHA256.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLCommon.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLCrackCommandLineData.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLGenCommandLineData.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLGenerateTable.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLGenerateTableMD5.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLGenerateTableNTLM.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLGenerateTableSHA1.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLGenerateTableSHA256.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLRegenerateChains.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLRegenerateChainsMD5.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLRegenerateChainsNTLM.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLRegenerateChainsSHA1.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLRegenerateChainsSHA256.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLUtils.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLKernels.cpp
    )

FILE( GLOB GRT_OPENCL_HOST_HEADERS ${CH_INCLUDE_DIR}/GRT_OpenCL_host/*.h )

ADD_LIBRARY ( ch_grt_opencl_host 
    ${GRT_OPENCL_HOST_SOURCES} 
    ${GRT_OPENCL_HOST_HEADERS} 
    )

ADD_DEPENDENCIES ( ch_grt_opencl_host
    grt_opencl_device_headers
    )

ADD_EXECUTABLE (
    GRTCrack-OpenCL
    ${grt_opencl_host_SOURCE_DIR}/GRTCLCrack.cpp
    )
TARGET_LINK_LIBRARIES ( GRTCrack-OpenCL
    ch_common
    ch_grt_common
    ch_grt_opencl_host
    ch_opencl_common
    argtable2
    ${OPENCL_LIBRARIES}
    ${CURL_LIBRARIES}
    ${CURSES_LIBRARIES}
    ${Boost_LIBRARIES}
    )

ADD_EXECUTABLE ( 
    GRTGen-OpenCL
    ${grt_opencl_host_SOURCE_DIR}/GRTCLGen.cpp
    )
TARGET_LINK_LIBRARIES ( GRTGen-OpenCL
    ch_opencl_common
    ch_grt_opencl_host
    ch_grt_common
    argtable2
    ${OPENCL_LIBRARIES}
    ${Boost_LIBRARIES}
    ${CURL_LIBRARIES}
    ${CH_THREAD_LIBRARIES}
    )

ADD_EXECUTABLE ( ShowConfig-OpenCL
    ${grt_opencl_host_SOURCE_DIR}/GRTCLShowConfiguration.cpp
    ${grt_opencl_host_SOURCE_DIR}/GRTCLUtils.cpp # for print_cl_errstring
    )
TARGET_LINK_LIBRARIES ( ShowConfig-OpenCL
    ch_opencl_common
    ${OPENCL_LIBRARIES}
    )

INSTALL ( TARGETS GRTCrack-OpenCL GRTGen-OpenCL ShowConfig-OpenCL 
          DESTINATION ${CMAKE_INSTALL_PREFIX} )
