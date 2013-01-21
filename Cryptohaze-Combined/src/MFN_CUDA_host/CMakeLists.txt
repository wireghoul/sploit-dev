PROJECT ( mfn_cuda_host )

SET ( MFN_CUDA_HOST_SOURCES 
    ${mfn_cuda_host_SOURCE_DIR}/MFNHashTypePlainCUDA.cpp
    ${mfn_cuda_host_SOURCE_DIR}/MFNHashTypePlainCUDA_DoubleMD5.cpp
    ${mfn_cuda_host_SOURCE_DIR}/MFNHashTypePlainCUDA_DupMD5.cpp
    ${mfn_cuda_host_SOURCE_DIR}/MFNHashTypePlainCUDA_LM.cpp
    ${mfn_cuda_host_SOURCE_DIR}/MFNHashTypePlainCUDA_MD5.cpp
    ${mfn_cuda_host_SOURCE_DIR}/MFNHashTypePlainCUDA_NTLM.cpp
    ${mfn_cuda_host_SOURCE_DIR}/MFNHashTypePlainCUDA_SHA1.cpp
    ${mfn_cuda_host_SOURCE_DIR}/MFNHashTypePlainCUDA_SHA256.cpp
    ${mfn_cuda_host_SOURCE_DIR}/MFNHashTypePlainCUDA_TripleMD5.cpp
    ${mfn_cuda_host_SOURCE_DIR}/MFNHashTypePlainCUDA_16HEX.cpp
    ${mfn_cuda_host_SOURCE_DIR}/MFNHashTypePlainCUDA_LOTUS.cpp
    ${mfn_cuda_host_SOURCE_DIR}/MFNHashTypeSaltedCUDA_MD5_PS.cpp
    ${mfn_cuda_host_SOURCE_DIR}/MFNHashTypeSaltedCUDA_IPBWL.cpp
    )


FILE( GLOB MFN_CUDA_HOST_HEADERS ${CH_INCLUDE_DIR}/MFN_CUDA_host/*.h )

ADD_LIBRARY ( ch_mfn_cuda_host 
    ${MFN_CUDA_HOST_SOURCES} 
    ${MFN_CUDA_HOST_HEADERS} 
    )
