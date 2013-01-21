PROJECT  ( grt_common )

SET ( GRT_COMMON_SOURCES
    ${grt_common_SOURCE_DIR}/GRTChainRunner.cpp
    ${grt_common_SOURCE_DIR}/GRTChainRunnerMD5.cpp
    ${grt_common_SOURCE_DIR}/GRTChainRunnerNTLM.cpp
    ${grt_common_SOURCE_DIR}/GRTChainRunnerSHA1.cpp
    ${grt_common_SOURCE_DIR}/GRTChainRunnerSHA256.cpp
    ${grt_common_SOURCE_DIR}/GRTCharsetSingle.cpp
    ${grt_common_SOURCE_DIR}/GRTCommon.cpp
    ${grt_common_SOURCE_DIR}/GRTCrackDisplayCurses.cpp
    ${grt_common_SOURCE_DIR}/GRTCrackDisplayDebug.cpp
    ${grt_common_SOURCE_DIR}/GRTHashes.cpp
    ${grt_common_SOURCE_DIR}/GRTTableHeaderV1.cpp
    ${grt_common_SOURCE_DIR}/GRTTableHeaderV2.cpp
    ${grt_common_SOURCE_DIR}/GRTTableHeaderV3.cpp
    ${grt_common_SOURCE_DIR}/GRTTableHeaderVWeb.cpp
    ${grt_common_SOURCE_DIR}/GRTTableSearchV1.cpp
    ${grt_common_SOURCE_DIR}/GRTTableSearchV2.cpp
    ${grt_common_SOURCE_DIR}/GRTTableSearchV3.cpp
    ${grt_common_SOURCE_DIR}/GRTTableSearchVWeb.cpp
    )

FILE( GLOB GRT_COMMON_HEADERS ${CH_INCLUDE_DIR}/GRT_Common/*.h )

ADD_LIBRARY ( ch_grt_common
    ${GRT_COMMON_SOURCES} 
    ${GRT_COMMON_HEADERS} 
    )

#IF ( WIN32 )
    SET_PROPERTY (
        TARGET ch_grt_common
        PROPERTY COMPILE_DEFINITIONS CURL_STATICLIB
        )
#ENDIF ( WIN32 )

ADD_EXECUTABLE (
    GRTAnalyze
    ${grt_common_SOURCE_DIR}/GRTAnalyzeTable.cpp
    )

# TODO - make pthread platform independent...
TARGET_LINK_LIBRARIES ( GRTAnalyze
    ch_grt_common
    ${Boost_LIBRARIES}
    ${CH_THREAD_LIBRARIES}
    )

ADD_EXECUTABLE (
    GRTBulkConvert
    ${grt_common_SOURCE_DIR}/GRTBulkConvert.cpp
    )
TARGET_LINK_LIBRARIES ( GRTBulkConvert
    ch_grt_common
    ${Boost_LIBRARIES}
    argtable2
    ${CH_THREAD_LIBRARIES}
    )


ADD_EXECUTABLE (
    GRTConvert
    ${grt_common_SOURCE_DIR}/GRTConvertTables.cpp
    )
TARGET_LINK_LIBRARIES ( GRTConvert
    ${Boost_LIBRARIES}
    ch_grt_common
    ${CH_THREAD_LIBRARIES}
    )


ADD_EXECUTABLE (
    GRTIndex
    ${grt_common_SOURCE_DIR}/GRTMakeIndex.cpp
    ${grt_common_SOURCE_DIR}/GRTMakeIndexCommandLineData.cpp
    )
TARGET_LINK_LIBRARIES ( GRTIndex
    ${Boost_LIBRARIES}
    ch_grt_common
    argtable2
    ${CH_THREAD_LIBRARIES}
    )


ADD_EXECUTABLE (
    GRTMakeTestFile
    ${grt_common_SOURCE_DIR}/GRTMakeTestFile.cpp
    )
TARGET_LINK_LIBRARIES ( GRTMakeTestFile
    ${Boost_LIBRARIES}
    ch_grt_common
    ${CH_THREAD_LIBRARIES}
    )


ADD_EXECUTABLE (
    GRTMerge
    ${grt_common_SOURCE_DIR}/GRTMerge.cpp
    ${grt_common_SOURCE_DIR}/GRTMergeCommandLineData.cpp
    )
TARGET_LINK_LIBRARIES ( GRTMerge
    ${Boost_LIBRARIES}
    ch_grt_common
    argtable2
    ${Boost_LIBRARIES}
    ${CH_THREAD_LIBRARIES}
    )


ADD_EXECUTABLE (
    GRTSearchTables
    ${grt_common_SOURCE_DIR}/GRTSearchTables.cpp
    )
TARGET_LINK_LIBRARIES ( GRTSearchTables
    ${Boost_LIBRARIES}
    ch_grt_common
    ${CH_THREAD_LIBRARIES}
    )

ADD_EXECUTABLE (
    GRTVerify
    ${grt_common_SOURCE_DIR}/GRTVerifyTable.cpp
    )
TARGET_LINK_LIBRARIES ( GRTVerify
    ${Boost_LIBRARIES}
    ch_grt_common
    ${Boost_LIBRARIES}
    ${CH_THREAD_LIBRARIES}
    )

ADD_EXECUTABLE (
    GRTVerifyImproved
    ${grt_common_SOURCE_DIR}/GRTVerifyTableImproved.cpp
    )
TARGET_LINK_LIBRARIES ( GRTVerifyImproved
    ch_grt_common
    ${Boost_LIBRARIES}
    argtable2
    ${CH_THREAD_LIBRARIES}
    )


INSTALL ( TARGETS 
            GRTAnalyze GRTBulkConvert GRTConvert GRTIndex GRTMakeTestFile 
            GRTMerge GRTSearchTables GRTVerify GRTVerifyImproved
          DESTINATION ${CMAKE_INSTALL_PREFIX} 
        )
