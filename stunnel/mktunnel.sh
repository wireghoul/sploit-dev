#!/bin/sh
if [ ! -e stunnel.pem ]; then
  echo "Generating new cert"
  openssl req -new -x509 -nodes -out stunnel.pem -keyout stunnel.pem
fi
if [ -z $2 ]; then
  echo "Usage: $0 listenport forwardip:port"
  exit 1
fi
echo "Creating config files to forward traffic between $1 and $2"
FWD='7777'
cat listen.in | sed -e"s/%LOCAL%/$1/" -e"s/%FWD%/127.0.0.1:$FWD/" > ssl-dec.conf 
cat forward.in | sed -e"s/%FWD%/$FWD/" -e"s/%REMOTE%/$2/" > ssl-enc.conf
echo "Starting services"
stunnel ./ssl-dec.conf
stunnel ./ssl-enc.conf
