PROJECT ( mfn_opencl_device )

# Defines macro for calling MakeHeaderFromFile.  SOURCE_NAME should be the name
# of the source file - ex input FileX.cl should just pass FileX with no 
# extension.  It will create a FileX_source.h in the same folder
MACRO ( MAKE_OPENCL_HEADER_FROM_FILE SOURCE_NAME )
    ADD_CUSTOM_COMMAND (
        OUTPUT ${CH_INCLUDE_DIR}/MFN_OpenCL_device/${SOURCE_NAME}_source.h
        COMMAND MakeHeaderFromFile
            ${mfn_opencl_device_SOURCE_DIR}/${SOURCE_NAME}.cl
            ${SOURCE_NAME}_source
            ${CH_INCLUDE_DIR}/MFN_OpenCL_device/${SOURCE_NAME}_source.h
        DEPENDS MakeHeaderFromFile 
                ${mfn_opencl_device_SOURCE_DIR}/${SOURCE_NAME}.cl
        )

    SET_SOURCE_FILES_PROPERTIES(
        ${CH_INCLUDE_DIR}/MFN_OpenCL_device/${SOURCE_NAME}_source.h
        PROPERTIES GENERATED TRUE
        )
ENDMACRO ( MAKE_OPENCL_HEADER_FROM_FILE SOURCE_NAME )

# Make the common function headers
MAKE_OPENCL_HEADER_FROM_FILE(MFN_OpenCL_Common)
MAKE_OPENCL_HEADER_FROM_FILE(MFN_OpenCL_MD5)
MAKE_OPENCL_HEADER_FROM_FILE(MFN_OpenCL_SHA256)
MAKE_OPENCL_HEADER_FROM_FILE(MFN_OpenCL_NTLM)
MAKE_OPENCL_HEADER_FROM_FILE(MFN_OpenCL_PasswordCopiers)
MAKE_OPENCL_HEADER_FROM_FILE(MFN_OpenCL_SaltUtilities)
MAKE_OPENCL_HEADER_FROM_FILE(MFN_OpenCL_BIN2HEX)

MAKE_OPENCL_HEADER_FROM_FILE(MFNHashTypePlainOpenCL_16HEX)
MAKE_OPENCL_HEADER_FROM_FILE(MFNHashTypePlainOpenCL_DoubleMD5)
MAKE_OPENCL_HEADER_FROM_FILE(MFNHashTypePlainOpenCL_DupMD5)
MAKE_OPENCL_HEADER_FROM_FILE(MFNHashTypePlainOpenCL_LOTUS)
MAKE_OPENCL_HEADER_FROM_FILE(MFNHashTypePlainOpenCL_MD5)
MAKE_OPENCL_HEADER_FROM_FILE(MFNHashTypePlainOpenCL_MD5WL)
MAKE_OPENCL_HEADER_FROM_FILE(MFNHashTypePlainOpenCL_NTLM)
MAKE_OPENCL_HEADER_FROM_FILE(MFNHashTypePlainOpenCL_NTLMWL)
MAKE_OPENCL_HEADER_FROM_FILE(MFNHashTypePlainOpenCL_SHA1)
MAKE_OPENCL_HEADER_FROM_FILE(MFNHashTypePlainOpenCL_SHA256)
MAKE_OPENCL_HEADER_FROM_FILE(MFNHashTypePlainOpenCL_DoubleSHA256)
MAKE_OPENCL_HEADER_FROM_FILE(MFNHashTypePlainOpenCL_SMD5)
MAKE_OPENCL_HEADER_FROM_FILE(MFNHashTypePlainOpenCL_SNTLM)
MAKE_OPENCL_HEADER_FROM_FILE(MFNHashTypeSaltedOpenCL_IPB)
MAKE_OPENCL_HEADER_FROM_FILE(MFNHashTypeSaltedOpenCL_IPBWL)
MAKE_OPENCL_HEADER_FROM_FILE(MFNHashTypeSaltedOpenCL_MD5_PS)
MAKE_OPENCL_HEADER_FROM_FILE(MFNHashTypeSaltedOpenCL_Phpass)
MAKE_OPENCL_HEADER_FROM_FILE(MFNHashTypeSaltedOpenCL_PhpassWL)

# the generated TRUE isn't carrying over to other projects so we just force the 
# headers to always get created with ALL DEPENDS
ADD_CUSTOM_TARGET ( mfn_opencl_device_headers echo "Creating MFN OpenCL Device Headers"
    DEPENDS
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFN_OpenCL_Common_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFN_OpenCL_MD5_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFN_OpenCL_SHA256_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFN_OpenCL_NTLM_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFN_OpenCL_PasswordCopiers_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFN_OpenCL_SaltUtilities_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFN_OpenCL_BIN2HEX_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFNHashTypePlainOpenCL_16HEX_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFNHashTypePlainOpenCL_DoubleMD5_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFNHashTypePlainOpenCL_DupMD5_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFNHashTypePlainOpenCL_LOTUS_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFNHashTypePlainOpenCL_MD5_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFNHashTypePlainOpenCL_MD5WL_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFNHashTypePlainOpenCL_NTLM_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFNHashTypePlainOpenCL_NTLMWL_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFNHashTypePlainOpenCL_SHA1_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFNHashTypePlainOpenCL_SHA256_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFNHashTypePlainOpenCL_DoubleSHA256_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFNHashTypePlainOpenCL_SMD5_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFNHashTypePlainOpenCL_SNTLM_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFNHashTypeSaltedOpenCL_IPB_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFNHashTypeSaltedOpenCL_IPBWL_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFNHashTypeSaltedOpenCL_MD5_PS_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFNHashTypeSaltedOpenCL_Phpass_source.h
    ${CH_INCLUDE_DIR}/MFN_OpenCL_device/MFNHashTypeSaltedOpenCL_PhpassWL_source.h
    )
ADD_DEPENDENCIES ( mfn_opencl_device_headers MakeHeaderFromFile )

#INSTALL (
#    FILES
#        ${mfn_opencl_device_SOURCE_DIR}/MFNHashTypePlainOpenCL_16HEX.cl
#        ${mfn_opencl_device_SOURCE_DIR}/MFNHashTypePlainOpenCL_MD5.cl
#        ${mfn_opencl_device_SOURCE_DIR}/MFNHashTypePlainOpenCL_MD5WL.cl
#        ${mfn_opencl_device_SOURCE_DIR}/MFNHashTypePlainOpenCL_SMD5.cl
#        ${mfn_opencl_device_SOURCE_DIR}/MFNHashTypePlainOpenCL_DupMD5.cl
#        ${mfn_opencl_device_SOURCE_DIR}/MFNHashTypePlainOpenCL_NTLM.cl
#        ${mfn_opencl_device_SOURCE_DIR}/MFNHashTypePlainOpenCL_NTLMWL.cl
#        ${mfn_opencl_device_SOURCE_DIR}/MFNHashTypePlainOpenCL_SHA1.cl
#        ${mfn_opencl_device_SOURCE_DIR}/MFNHashTypePlainOpenCL_SHA256.cl
#        ${mfn_opencl_device_SOURCE_DIR}/MFNHashTypeSaltedOpenCL_MD5_PS.cl
#        ${mfn_opencl_device_SOURCE_DIR}/MFNHashTypeSaltedOpenCL_IPB.cl
#        ${mfn_opencl_device_SOURCE_DIR}/MFNHashTypePlainOpenCL_LOTUS.cl
#    DESTINATION ${CMAKE_INSTALL_PREFIX}/src/MFN_OpenCL_device/
#    )
