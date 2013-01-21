PROJECT ( mf_cuda_device )

SET ( MF_CUDA_DEVICE_SOURCES 
    ${mf_cuda_device_SOURCE_DIR}/CHCommandLineData.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashType.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypeMSSQL.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypeMSSQL_CPU.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlain.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainDoubleMD5.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainDuplicatedMD5.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainDuplicatedNTLM.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainLM.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainMD4.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainMD5.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainMD5ofSHA1.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainMD5Single.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainMySQL323.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainNTLM.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainSHA1.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainSHA1ofMD5.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainSHA256.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypePlainTripleMD5.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypeSalted.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypeSaltedDCC.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypeSaltedMD5PassSalt.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypeSaltedMD5SaltPass.cpp
    ${mf_cuda_device_SOURCE_DIR}/CHHashTypeSaltedSSHA.cpp
    )


FILE( GLOB MF_CUDA_DEVICE_HEADERS ${CH_INCLUDE_DIR}/Multiforcer_CUDA_device/*.h )

CUDA_ADD_LIBRARY ( ch_mf_cuda_host 
    ${MF_CUDA_DEVICE_SOURCES}
    ${MF_CUDA_DEVICE_HEADERS} 
    )
