PROJECT ( mf_common )

SET ( MF_COMMON_SOURCES 
    ${mf_common_SOURCE_DIR}/CHCommon.cpp
    ${mf_common_SOURCE_DIR}/CHDisplayDaemon.cpp
    ${mf_common_SOURCE_DIR}/CHDisplayDebug.cpp
    ${mf_common_SOURCE_DIR}/CHDisplayNcurses.cpp
    ${mf_common_SOURCE_DIR}/CHHashes.cpp
    ${mf_common_SOURCE_DIR}/CHNetworkClient.cpp
    ${mf_common_SOURCE_DIR}/CHNetworkServer.cpp
    )

FILE( GLOB MF_COMMON_HEADERS ${CH_INCLUDE_DIR}/MF_Common/*.h )

ADD_LIBRARY ( ch_mf_common
    ${MF_COMMON_SOURCES}
    ${MF_COMMON_HEADERS}
    ) 

SET ( MF_NETWORK_SOURCES 
    ${mf_common_SOURCE_DIR}/CHNetworkClient.cpp
    ${mf_common_SOURCE_DIR}/CHNetworkServer.cpp
    )
ADD_LIBRARY ( ch_network_class
    ${MF_NETWORK_SOURCES}
    ${MF_COMMON_HEADERS}
    ) 


ADD_EXECUTABLE ( Cryptohaze-Multiforcer
    ${mf_common_SOURCE_DIR}/CHMain.cpp
    ${MF_COMMON_SOURCES}
    )
TARGET_LINK_LIBRARIES ( Cryptohaze-Multiforcer
    ch_mf_common
    ch_common
    ch_mf_cuda_host
    ch_mf_cuda_device 
    argtable2
    ${Boost_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CURSES_LIBRARIES}
    )

INSTALL ( TARGETS Cryptohaze-Multiforcer 
          DESTINATION ${CMAKE_INSTALL_PREFIX} 
        )
