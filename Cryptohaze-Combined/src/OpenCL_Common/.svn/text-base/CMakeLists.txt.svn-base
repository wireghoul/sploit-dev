PROJECT ( opencl_common )

SET ( OPENCL_COMMON_SOURCES
    ${opencl_common_SOURCE_DIR}/GRTOpenCL.cpp
    ${opencl_common_SOURCE_DIR}/BFIPatcher.cpp
    )

FILE( GLOB OPENCL_COMMON_HEADERS ${CH_INCLUDE_DIR}/OpenCL_Common/*.h )

ADD_LIBRARY ( ch_opencl_common 
    ${OPENCL_COMMON_SOURCES} 
    ${OPENCL_COMMON_HEADERS}
    )

TARGET_LINK_LIBRARIES ( 
    ch_opencl_common 
    )

ADD_EXECUTABLE ( MakeHeaderFromFile 
    ${opencl_common_SOURCE_DIR}/MakeHeaderFromFile.cpp
    )
