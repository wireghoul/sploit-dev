PROJECT ( mf_cuda_device )

SET ( MF_CUDA_DEVICE_SOURCES
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypeMSSQL.cu
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainDoubleMD5.cu
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainDuplicatedMD5.cu
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainDuplicatedNTLM.cu
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainLM.cu
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainMD4.cu
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainMD5.cu
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainMD5ofSHA1.cu
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainMD5Single.cu
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainMySQL323.cu
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainNTLM.cu
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainSHA1.cu
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainSHA1ofMD5.cu
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainSHA256.cu
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainTripleMD5.cu
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypeSaltedDCC.cu
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypeSaltedMD5PassSalt.cu
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypeSaltedMD5SaltPass.cu
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypeSaltedSSHA.cu
    )

FILE( GLOB MF_CUDA_DEVICE_HEADERS ${CH_INCLUDE_DIR}/Multiforcer_CUDA_device/*.h )

CUDA_ADD_LIBRARY ( ch_mf_cuda_device 
    ${MF_CUDA_DEVICE_SOURCES} 
    ${MF_CUDA_DEVICE_HEADERS} 
    )
