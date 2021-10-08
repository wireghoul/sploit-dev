#!/bin/bash
# cleanup
mkdir -p /tmp/arginjecttest
cd /tmp/arginjecttest
rm -rf -- aaaa bbbb cccc --help
mainpid=$$
(sleep 5; kill $mainpid) &
wdog=$!

# Clear the terminal line
echo -en "\r                                                                         "
echo -en "\rChecking $1 for arg injection"
$1 -- aaaa bbbb   > /tmp/argi1 2>&1
$1 -- cccc --help > /tmp/argi2 2>&1

if [[ $(wc -l /tmp/argi1|cut -d' ' -f1) != $(wc -l /tmp/argi2|cut -d' ' -f1) ]]; then
    echo -e " ... Potentially vulnerable\n [+] PoC: $1 -- cccc --help\n"
fi

cd - >/dev/null
kill $wdog
