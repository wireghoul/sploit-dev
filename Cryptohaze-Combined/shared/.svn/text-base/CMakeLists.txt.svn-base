PROJECT ( ch_shared )

FILE ( GLOB charsets ${ch_shared_SOURCE_DIR}/charsets/* )
FILE ( GLOB charsets_ips ${ch_shared_SOURCE_DIR}/charsets/ip_addresses* )
FILE ( GLOB test_hashes ${ch_shared_SOURCE_DIR}/test_hashes/* )

INSTALL (
    DIRECTORY charsets test_hashes
    DESTINATION ${CMAKE_INSTALL_PREFIX}
)

