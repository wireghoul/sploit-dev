PROJECT ( grt_cuda_host )

SET ( GRT_CUDA_HOST_SOURCES 
    ${grt_cuda_host_SOURCE_DIR}/GRTCandidateHashes.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTCandidateHashesMD5.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTCandidateHashesNTLM.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTCandidateHashesSHA1.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTCandidateHashesSHA256.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTCrackCommandLineData.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTGenCommandLineData.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTGenerateTable.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTGenerateTableDuplicatedMD5.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTGenerateTableDuplicatedNTLM.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTGenerateTableMD5.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTGenerateTableNTLM.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTGenerateTableSHA1.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTGenerateTableSHA256.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTRegenerateChains.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTRegenerateChainsMD5.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTRegenerateChainsNTLM.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTRegenerateChainsSHA1.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTRegenerateChainsSHA256.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTWebGenCommandLineData.cpp
    ${grt_cuda_host_SOURCE_DIR}/GRTWebGen.cpp
    )


FILE( GLOB GRT_CUDA_HOST_HEADERS ${CH_INCLUDE_DIR}/GRT_CUDA_host/*.h )

ADD_LIBRARY ( ch_grt_cuda_host 
    ${GRT_CUDA_HOST_SOURCES} 
    ${GRT_CUDA_HOST_HEADERS} 
    )


ADD_EXECUTABLE ( GRTCrack-CUDA
    ${grt_cuda_host_SOURCE_DIR}/GRTCrack.cpp
    )
TARGET_LINK_LIBRARIES ( GRTCrack-CUDA
    ch_common
    ch_grt_cuda_host
    ch_grt_cuda_device
    ch_grt_common
    ch_cuda_common
    ch_timer
    argtable2
    ${CURSES_LIBRARIES}
    ${Boost_LIBRARIES}
    ${CURL_LIBRARIES}
    ${CUDA_SDK_LIBRARIES}
    )

ADD_EXECUTABLE ( GRTGen-CUDA
    ${grt_cuda_host_SOURCE_DIR}/GRTGen.cpp
    )
TARGET_LINK_LIBRARIES ( GRTGen-CUDA
    ch_common
    ch_grt_cuda_host
    ch_grt_cuda_device
    ch_cuda_common
    ch_grt_common
    ch_mf_common
    ch_timer
    argtable2
    ${Boost_LIBRARIES}
    ${CURL_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_SDK_LIBRARIES}
    )

INSTALL ( TARGETS GRTCrack-CUDA GRTGen-CUDA
          DESTINATION ${CMAKE_INSTALL_PREFIX} 
        )
