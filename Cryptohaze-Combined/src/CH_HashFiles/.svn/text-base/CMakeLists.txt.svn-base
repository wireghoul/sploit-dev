PROJECT ( mfn_hashfiles )

SET ( MFN_HASHFILES_SOURCES 
    ${mfn_hashfiles_SOURCE_DIR}/CHHashFileV.cpp
    ${mfn_hashfiles_SOURCE_DIR}/CHHashFileVIKE.cpp
    ${mfn_hashfiles_SOURCE_DIR}/CHHashFileVLM.cpp
    ${mfn_hashfiles_SOURCE_DIR}/CHHashFileVPlain.cpp
    ${mfn_hashfiles_SOURCE_DIR}/CHHashFileVPlain.pb.cpp
    ${mfn_hashfiles_SOURCE_DIR}/CHHashFileVSalted.cpp
    ${mfn_hashfiles_SOURCE_DIR}/CHHashFileVSalted.pb.cpp
    ${mfn_hashfiles_SOURCE_DIR}/CHHashFileVPhpass.cpp
    )


FILE( GLOB MFN_HASHFILES_HEADERS ${CH_INCLUDE_DIR}/CH_HashFiles/*.h )

ADD_LIBRARY ( mfn_hashfiles 
    ${MFN_HASHFILES_SOURCES} 
    ${MFN_HASHFILES_HEADERS} 
    )

