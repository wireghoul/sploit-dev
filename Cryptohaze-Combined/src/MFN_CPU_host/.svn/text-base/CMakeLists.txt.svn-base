PROJECT ( mfn_cpu_host )

SET ( MFN_CPU_HOST_SOURCES 
    ${mfn_cpu_host_SOURCE_DIR}/MFNHashTypePlainCPU.cpp
    ${mfn_cpu_host_SOURCE_DIR}/MFNHashTypePlainCPU_MD5.cpp
    ${mfn_cpu_host_SOURCE_DIR}/MFNHashTypePlainCPU_NTLM.cpp
    )


FILE( GLOB MFN_CPU_HOST_HEADERS ${CH_INCLUDE_DIR}/MFN_CPU_host/*.h )

ADD_LIBRARY ( ch_mfn_cpu_host 
    ${MFN_CPU_HOST_SOURCES} 
    ${MFN_CPU_HOST_HEADERS} 
    )

IF (NOT MSVC)
    set_target_properties(ch_mfn_cpu_host
        PROPERTIES COMPILE_FLAGS "-msse2 -O3")
ELSE(NOT MSVC)
    set_target_properties(ch_mfn_cpu_host
        PROPERTIES COMPILE_FLAGS "/arch:SSE2")
ENDIF(NOT MSVC)