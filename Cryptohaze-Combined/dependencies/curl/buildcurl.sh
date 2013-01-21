#!/bin/bash

./configure --disable-debug --enable-optimize --disable-curldebug --disable-ares --disable-largefile --enable-http --disable-ftp --disable-file --disable-ldap --disable-ldaps --disable-rtsp --enable-proxy --disable-dict --disable-telnet --disable-tftp --disable-pop3 --disable-imap --disable-smtp --disable-gopher --disable-manual --disable-sspi --disable-crypto-auth --disable-ntlm-wb --disable-tls-srp --disable-cookies --disable-hidden-symbols --without-ssl --without-krb4 --without-gssapi --without-zlib --without-gnutls --without-polarssl --without-cyassl --without-axtls --without-ca-bundle --without-ca-path --without-libssh2 --without-librtmp --without-libidn

make -j5

echo Libs should be in libs/.libs
