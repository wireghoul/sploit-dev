PROJECT ( cuda_common )

SET ( CUDA_COMMON_SOURCES
    ${cuda_common_SOURCE_DIR}/CHCudaUtils.cpp
    ${cuda_common_SOURCE_DIR}/CUDA_Common.cpp
    )

FILE( GLOB CUDA_COMMON_HEADERS ${CH_INCLUDE_DIR}/CH_Common/*.h )

ADD_LIBRARY ( ch_cuda_common 
    ${CUDA_COMMON_SOURCES} 
    ${CUDA_COMMON_HEADERS} 
    )
