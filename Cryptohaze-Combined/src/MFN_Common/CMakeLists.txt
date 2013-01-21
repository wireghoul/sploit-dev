PROJECT ( mfn_common )

SET ( MFN_COMMON_SOURCES 
    ${mfn_common_SOURCE_DIR}/MFNCommandLineData.cpp
    ${mfn_common_SOURCE_DIR}/MFNDisplay.cpp
    ${mfn_common_SOURCE_DIR}/MFNDisplayCurses.cpp
    ${mfn_common_SOURCE_DIR}/MFNDisplayDaemon.cpp
    ${mfn_common_SOURCE_DIR}/MFNDisplayDebug.cpp
    ${mfn_common_SOURCE_DIR}/MFNHashClassLauncher.cpp
    ${mfn_common_SOURCE_DIR}/MFNHashIdentifiers.cpp
    ${mfn_common_SOURCE_DIR}/MFNHashType.cpp
    ${mfn_common_SOURCE_DIR}/MFNHashTypePlain.cpp
    ${mfn_common_SOURCE_DIR}/MFNMultiforcerClassFactory.cpp


    ${mfn_common_SOURCE_DIR}/MFNCharsetNew.pb.cpp
    ${mfn_common_SOURCE_DIR}/MFNGeneralInformation.cpp
    ${mfn_common_SOURCE_DIR}/MFNNetworkClient.cpp
    ${mfn_common_SOURCE_DIR}/MFNNetworkCommon.cpp
    ${mfn_common_SOURCE_DIR}/MFNNetworkRPC.pb.cpp
    ${mfn_common_SOURCE_DIR}/MFNNetworkServer.cpp
    ${mfn_common_SOURCE_DIR}/MFNPermutations.cpp
    ${mfn_common_SOURCE_DIR}/MFNWorkunitNetwork.cpp
    ${mfn_common_SOURCE_DIR}/MFNWorkunitRobust.cpp
    ${mfn_common_SOURCE_DIR}/MFNWorkunitWordlist.cpp
    ${mfn_common_SOURCE_DIR}/MFNWorkunit.pb.cpp
    )

FILE ( GLOB MFN_COMMON_HEADERS ${CH_INCLUDE_DIR}/MFN_Common/*.h )

# We only want to build MFNDisplayGui if QT is found so remove the header
LIST ( REMOVE_ITEM MFN_COMMON_HEADERS ${CH_INCLUDE_DIR}/MFN_Common/MFNDisplayGui.h )
LIST ( REMOVE_ITEM MFN_COMMON_HEADERS ${CH_INCLUDE_DIR}/MFN_Common/MFNDisplayGuiSettings.h )

ADD_LIBRARY ( ch_mfn_common
    ${MFN_COMMON_SOURCES}
    ${MFN_COMMON_HEADERS}
    ) 

ADD_EXECUTABLE ( New-Multiforcer
    ${mfn_common_SOURCE_DIR}/MFNMain.cpp
    ${mfn_common_SOURCE_DIR}/MFNDisplayCliMain.cpp
    ${MFN_COMMON_SOURCES}
    )
TARGET_LINK_LIBRARIES ( New-Multiforcer
    ch_common
    ch_mfn_common
    ch_mfn_cpu_host
    ch_mfn_cuda_host
    ch_mfn_cuda_device 
    ch_mfn_opencl_host
    ch_opencl_common 
    ch_network_class
    argtable2
    ch_cuda_common
    ch_network_workunit
    mfn_hashfiles
    ch_hash_implementation
    ${Boost_LIBRARIES}
    ${OPENCL_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CURSES_LIBRARIES}
    ${PROTOBUF_LIBRARY}
    )

INSTALL ( TARGETS New-Multiforcer
          DESTINATION ${CMAKE_INSTALL_PREFIX} 
        )

# if this fails you can help it by specifying QTDIR env var to point to the root
# dir of your QT install, ex D:\dev\QtSDK\Desktop\Qt\4.8.1\msvc2010
FIND_PACKAGE( Qt4 4.8 COMPONENTS QtCore QtGui )
IF ( WIN32 AND QT4_FOUND )
    MESSAGE ( STATUS "Found QT4 - Building systray GUI" )

    INCLUDE ( ${QT_USE_FILE} )

    SET ( CH_MFN_GUI_SOURCES
        ${mfn_common_SOURCE_DIR}/MFNGuiMain.cpp 
        ${mfn_common_SOURCE_DIR}/MFNDisplayGui.cpp 
        ${mfn_common_SOURCE_DIR}/MFNDisplayGuiSettings.cpp
        )

    SET ( CH_MFN_GUI_HEADERS 
        ${CH_INCLUDE_DIR}/MFN_Common/MFNDisplayGui.h 
        ${CH_INCLUDE_DIR}/MFN_Common/MFNDisplayGuiSettings.h 
        )

    QT4_WRAP_CPP( CH_MFN_QT_MOC ${CH_MFN_GUI_HEADERS} )

    QT4_WRAP_UI ( CH_MFN_QT_UI
        ${mfn_common_SOURCE_DIR}/MFNDisplayGui.ui
        ${mfn_common_SOURCE_DIR}/MFNDisplayGuiSettings.ui
        )

    QT4_ADD_RESOURCES ( CH_MFN_QT_RC
        ${mfn_common_SOURCE_DIR}/MFNDisplayGui.qrc
        )

    # the UI headers get created in this project's build dir
    INCLUDE_DIRECTORIES ( ${mfn_common_BINARY_DIR} )

    ADD_EXECUTABLE ( MFN_Gui 
        WIN32                   # specifying removes cmd.exe from opening
        ${mfn_common_SOURCE_DIR}/MFNMain.cpp
        ${MFN_COMMON_SOURCES}
        ${CH_MFN_GUI_SOURCES}    
        ${CH_MFN_GUI_HEADERS}
        ${CH_MFN_QT_MOC}
        ${CH_MFN_QT_UI}
        ${CH_MFN_QT_RC}
        )
    TARGET_LINK_LIBRARIES ( MFN_Gui 
        ch_common

        ch_mfn_common
        ch_mfn_cpu_host
        ch_mfn_cuda_host
        ch_mfn_cuda_device 
        ch_mfn_opencl_host
        ch_opencl_common 
        ch_network_class
        argtable2
        ch_cuda_common
        ch_network_workunit
        mfn_hashfiles
        ${Boost_LIBRARIES}
        ${OPENCL_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${CURSES_LIBRARIES}
        ${PROTOBUF_LIBRARY}

        ${QT_LIBRARIES} 
        )
ENDIF ( WIN32 AND QT4_FOUND )
