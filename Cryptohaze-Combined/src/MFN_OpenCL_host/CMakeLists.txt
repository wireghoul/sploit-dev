PROJECT ( mfn_opencl_host )

SET ( MFN_OPENCL_HOST_SOURCES 
    ${mfn_opencl_host_SOURCE_DIR}/MFNHashTypePlainOpenCL.cpp
    ${mfn_opencl_host_SOURCE_DIR}/MFNHashTypePlainOpenCL_MD5.cpp
    ${mfn_opencl_host_SOURCE_DIR}/MFNHashTypePlainOpenCL_MD5WL.cpp
    ${mfn_opencl_host_SOURCE_DIR}/MFNHashTypePlainOpenCL_SMD5.cpp
    ${mfn_opencl_host_SOURCE_DIR}/MFNHashTypePlainOpenCL_DupMD5.cpp
    ${mfn_opencl_host_SOURCE_DIR}/MFNHashTypePlainOpenCL_DoubleMD5.cpp
    ${mfn_opencl_host_SOURCE_DIR}/MFNHashTypePlainOpenCL_NTLM.cpp
    ${mfn_opencl_host_SOURCE_DIR}/MFNHashTypePlainOpenCL_NTLMWL.cpp
    ${mfn_opencl_host_SOURCE_DIR}/MFNHashTypePlainOpenCL_SNTLM.cpp
    ${mfn_opencl_host_SOURCE_DIR}/MFNHashTypePlainOpenCL_SHA1.cpp
    ${mfn_opencl_host_SOURCE_DIR}/MFNHashTypePlainOpenCL_SHA256.cpp
    ${mfn_opencl_host_SOURCE_DIR}/MFNHashTypePlainOpenCL_DoubleSHA256.cpp
    ${mfn_opencl_host_SOURCE_DIR}/MFNHashTypeSaltedOpenCL_MD5_PS.cpp
    ${mfn_opencl_host_SOURCE_DIR}/MFNHashTypeSaltedOpenCL_IPB.cpp
    ${mfn_opencl_host_SOURCE_DIR}/MFNHashTypeSaltedOpenCL_IPBWL.cpp
    ${mfn_opencl_host_SOURCE_DIR}/MFNHashTypePlainOpenCL_LOTUS.cpp
    ${mfn_opencl_host_SOURCE_DIR}/MFNOpenCLMetaprograms.cpp
    ${mfn_opencl_host_SOURCE_DIR}/MFNHashTypeSaltedOpenCL_Phpass.cpp
    ${mfn_opencl_host_SOURCE_DIR}/MFNHashTypeSaltedOpenCL_PhpassWL.cpp
    ${mfn_opencl_host_SOURCE_DIR}/MFNKernels.cpp
    )

FILE( GLOB MFN_OPENCL_HOST_HEADERS ${CH_INCLUDE_DIR}/MFN_OpenCL_host/*.h )

ADD_LIBRARY ( ch_mfn_opencl_host 
    ${MFN_OPENCL_HOST_SOURCES} 
    ${MFN_OPENCL_HOST_HEADERS} 
    )


ADD_DEPENDENCIES ( ch_mfn_opencl_host 
    mfn_opencl_device_headers
    )

