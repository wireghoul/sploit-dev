PROJECT ( ch_common )
SET ( CH_COMMON_SOURCES 
    ${ch_common_SOURCE_DIR}/b64.cpp
    ${ch_common_SOURCE_DIR}/CHCharsetMulti.cpp
    ${ch_common_SOURCE_DIR}/CHCharsetNew.cpp
    ${ch_common_SOURCE_DIR}/CHCharsetSingle.cpp
    ${ch_common_SOURCE_DIR}/CHHashFileLM.cpp
    ${ch_common_SOURCE_DIR}/CHHashFileMSSQL.cpp
    ${ch_common_SOURCE_DIR}/CHHashFilePlain32.cpp
    ${ch_common_SOURCE_DIR}/CHHashFilePlainSHA.cpp
    ${ch_common_SOURCE_DIR}/CHHashFileSalted32.cpp
    ${ch_common_SOURCE_DIR}/CHHashFileSaltedIKE.cpp
    ${ch_common_SOURCE_DIR}/CHHashFileSaltedSSHA.cpp
    ${ch_common_SOURCE_DIR}/CHHiresTimer.cpp
    ${ch_common_SOURCE_DIR}/CHWorkunit.cpp
    ${ch_common_SOURCE_DIR}/CHWorkunitNetwork.cpp
    ${ch_common_SOURCE_DIR}/CHWorkunitRobust.cpp
    ${ch_common_SOURCE_DIR}/CMakeLists.txt
    ${ch_common_SOURCE_DIR}/GRTHashFilePlain.cpp
    ${ch_common_SOURCE_DIR}/GRTWorkunit.cpp
    ${ch_common_SOURCE_DIR}/mtwist.cpp
)


SET ( CH_HASH_IMPLEMENTATION_SOURCES 
    ${ch_common_SOURCE_DIR}/CHHashImplementation.cpp
    ${ch_common_SOURCE_DIR}/CHHashImplementationMD5.cpp
    ${ch_common_SOURCE_DIR}/CHHashImplementationNTLM.cpp
    ${ch_common_SOURCE_DIR}/CHHashImplementationSHA1.cpp
    ${ch_common_SOURCE_DIR}/CHHashImplementationSHA256.cpp
)

FILE( GLOB CH_COMMON_HEADERS ${CH_INCLUDE_DIR}/CH_Common/*.h )

ADD_LIBRARY ( ch_hash_implementation
    ${CH_HASH_IMPLEMENTATION_SOURCES} 
    ${CH_COMMON_HEADERS} 
    )


IF (NOT MSVC)
    set_target_properties(ch_hash_implementation
        PROPERTIES COMPILE_FLAGS "-O3")
ENDIF(NOT MSVC)

ADD_LIBRARY ( ch_common
    ${CH_COMMON_SOURCES} 
    ${CH_COMMON_HEADERS} 
    )

ADD_LIBRARY ( ch_timer
    ${ch_common_SOURCE_DIR}/CHHiresTimer.cpp
    ${CH_INCLUDE_DIR}/CH_Common/CHHiresTimer.h
)

TARGET_LINK_LIBRARIES ( ch_common ch_grt_common ch_hash_implementation )

SET ( CH_NETWORK_WORKUNIT_SOURCES 
    ${ch_common_SOURCE_DIR}/CHWorkunitNetwork.cpp
)

ADD_LIBRARY ( ch_network_workunit 
    ${CH_NETWORK_WORKUNIT_SOURCES} 
    ${CH_COMMON_HEADERS} 
    )