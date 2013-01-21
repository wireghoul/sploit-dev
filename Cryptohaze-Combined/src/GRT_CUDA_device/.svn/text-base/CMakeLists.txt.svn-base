PROJECT ( grt_cuda_device )

SET ( GRT_CUDA_DEVICE_SOURCES 
    ${grt_cuda_device_SOURCE_DIR}/GRTCandidateHashesMD5.cu
    ${grt_cuda_device_SOURCE_DIR}/GRTCandidateHashesNTLM.cu
    ${grt_cuda_device_SOURCE_DIR}/GRTCandidateHashesSHA1.cu
    ${grt_cuda_device_SOURCE_DIR}/GRTCandidateHashesSHA256.cu
    ${grt_cuda_device_SOURCE_DIR}/GRTGenerateTableDuplicatedMD5.cu
    ${grt_cuda_device_SOURCE_DIR}/GRTGenerateTableDuplicatedNTLM.cu
    ${grt_cuda_device_SOURCE_DIR}/GRTGenerateTableMD5.cu
    ${grt_cuda_device_SOURCE_DIR}/GRTGenerateTableNTLM.cu
    ${grt_cuda_device_SOURCE_DIR}/GRTGenerateTableSHA1.cu
    ${grt_cuda_device_SOURCE_DIR}/GRTGenerateTableSHA256.cu
    ${grt_cuda_device_SOURCE_DIR}/GRTRegenerateChainsMD5.cu
    ${grt_cuda_device_SOURCE_DIR}/GRTRegenerateChainsNTLM.cu
    ${grt_cuda_device_SOURCE_DIR}/GRTRegenerateChainsSHA1.cu
    ${grt_cuda_device_SOURCE_DIR}/GRTRegenerateChainsSHA256.cu
    )

FILE( GLOB GRT_CUDA_DEVICE_HEADERS ${CH_INCLUDE_DIR}/GRT_CUDA_device/*.h )

CUDA_ADD_LIBRARY ( ch_grt_cuda_device 
    ${GRT_CUDA_DEVICE_SOURCES} 
    ${GRT_CUDA_DEVICE_HEADERS} 
    )
