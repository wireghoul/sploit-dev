PROJECT ( pdcurs34 )

FILE ( GLOB pdcurs34_SOURCES ${pdcurs34_SOURCE_DIR}/pdcurses/*.c ${pdcurs34_SOURCE_DIR}/win32/*.c )
FILE ( GLOB pdcurs34_HEADERS ${pdcurs34_SOURCE_DIR}/*.h )

ADD_LIBRARY ( pdcurs34 STATIC
    ${pdcurs34_SOURCES}
    ${pdcurs34_HEADERS}
    )

INCLUDE_DIRECTORIES ( ${pdcurs34_SOURCE_DIR} )

