PROJECT ( mfn_cuda_device )

SET ( MFN_CUDA_DEVICE_SOURCES 
    ${mfn_cuda_device_SOURCE_DIR}/MFNHashTypePlainCUDA_DoubleMD5.cu
    ${mfn_cuda_device_SOURCE_DIR}/MFNHashTypePlainCUDA_DupMD5.cu
    ${mfn_cuda_device_SOURCE_DIR}/MFNHashTypePlainCUDA_LM.cu
    ${mfn_cuda_device_SOURCE_DIR}/MFNHashTypePlainCUDA_MD5.cu
    ${mfn_cuda_device_SOURCE_DIR}/MFNHashTypePlainCUDA_NTLM.cu
    ${mfn_cuda_device_SOURCE_DIR}/MFNHashTypePlainCUDA_SHA1.cu
    ${mfn_cuda_device_SOURCE_DIR}/MFNHashTypePlainCUDA_SHA256.cu
    ${mfn_cuda_device_SOURCE_DIR}/MFNHashTypePlainCUDA_TripleMD5.cu
    ${mfn_cuda_device_SOURCE_DIR}/MFNHashTypeSaltedCUDA_MD5_PS.cu
    ${mfn_cuda_device_SOURCE_DIR}/MFNHashTypeSaltedCUDA_IPBWL.cu
    ${mfn_cuda_device_SOURCE_DIR}/MFNHashTypePlainCUDA_16HEX.cu
    ${mfn_cuda_device_SOURCE_DIR}/MFNHashTypePlainCUDA_LOTUS.cu
    )

FILE( GLOB MFN_CUDA_DEVICE_HEADERS ${CH_INCLUDE_DIR}/MFN_CUDA_device/*.h )

CUDA_ADD_LIBRARY ( ch_mfn_cuda_device 
    ${MFN_CUDA_DEVICE_SOURCES} 
    ${MFN_CUDA_DEVICE_HEADERS} 
    )
