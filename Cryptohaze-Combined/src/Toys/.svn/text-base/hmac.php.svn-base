<?php

function hmac_sha1($key, $data)
{
    // Adjust key to exactly 64 bytes
    if (strlen($key) > 64) {
        $key = str_pad(sha1($key, true), 64, chr(0));
    }
    if (strlen($key) < 64) {
        $key = str_pad($key, 64, chr(0));
    }

    // Outter and Inner pad
    $opad = str_repeat(chr(0x5C), 64);
    $ipad = str_repeat(chr(0x36), 64);

    // Xor key with opad & ipad
    for ($i = 0; $i < strlen($key); $i++) {
        $opad[$i] = $opad[$i] ^ $key[$i];
        $ipad[$i] = $ipad[$i] ^ $key[$i];
    }

    return sha1($opad.sha1($ipad.$data, true));
}

function hmac_md5($key, $data)
{
    // Adjust key to exactly 64 bytes
    if (strlen($key) > 64) {
        $key = str_pad(sha1($key, true), 64, chr(0));
    }
    if (strlen($key) < 64) {
        $key = str_pad($key, 64, chr(0));
    }

    // Outter and Inner pad
    $opad = str_repeat(chr(0x5C), 64);
    $ipad = str_repeat(chr(0x36), 64);

    // Xor key with opad & ipad
    for ($i = 0; $i < strlen($key); $i++) {
        $opad[$i] = $opad[$i] ^ $key[$i];
        $ipad[$i] = $ipad[$i] ^ $key[$i];
    }
    print "md5 of ipad.data: " . md5($ipad.$data) . "\n";
    print "md5 of full hash: " . md5($opad.md5($ipad.$data, true)) . "\n";
    print "md5 of just opad: " . md5($opad) . "\n";

    return md5($opad.md5($ipad.$data, true));
}

hmac_md5("key", "The quick brown fox jumps over the lazy dog");
