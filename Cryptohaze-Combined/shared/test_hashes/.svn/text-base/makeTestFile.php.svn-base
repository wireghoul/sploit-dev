<?
/*
 * Generic generator of hashes for the Cryptohaze tool testing.  This file is 
 * placed in the public domain and can be used by anyone.  Please keep the
 * copyright if you modify it, and please submit changes back to me!
 * 
 * Copyright 2012 Bitweasil <bitweasil@gmail.com>
 */

/**
 * Usage: makeTestFile [hash name] [number passwords] [pass length]
 *   [opt: max salt length] [opt: charset flags]
 * 
 * [hash name]: The hame of the hash to use (MD5, NTLM, etc)
 * [number passwords]: How many password hashes to generate.
 * [pass length]: As stated...
 * [max salt length]: For salted hashes, the maximum salt length to use.  Since
 *   most of the kernels handle varied salt lengths, this is used to test a
 *   robust selection of salt lengths.
 * [charset flags]: Hashcat-style charset flags (?u?l?d?s) - space is part of ?s
 */

// Check for sane argv count
if (($argc < 4) || ($argc > 6)) {
    print "Usage: makeTestFile.php [hash name] [number passwords] [pass length]\n";
    print "  [opt: max salt length] [opt: charset flags]\n";

    // Please keep this updated...
    print "Supported hashes: MD5 NTLM SHA1 SHA256 MD5_PS IPB PHPASS\n";
    exit;
}

$ipbSaltCharset = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~";



// If the charset flags are specified, do something sane with them.
if ($argc == 6) {
    $charset = "";
    // Parse argv[5] and create the charset.
    if (strpos($argv[5], "?u") !== FALSE) {
        $charset .= "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    }
    if (strpos($argv[5], "?l") !== FALSE) {
        $charset .= "abcdefghijklmnopqrstuvwxyz";
    }
    if (strpos($argv[5], "?d") !== FALSE) {
        $charset .= "0123456789";
    }
    if (strpos($argv[5], "?s") !== FALSE) {
        $charset .= " !\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~";
    }
    if (strpos($argv[5], "?1") !== FALSE) {
        $charset .= "1";
    }
    if (strpos($argv[5], "?2") !== FALSE) {
        $charset .= "2";
    }
    if (strlen($charset) == 0) {
        print "No charset specified!\n";
        exit;
    }
} else {
    // Default charset - everything.
    $charset = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~";
}

switch ($argv[1]) {
    case "MD5":
        generateMD5TestHashes($argv[2], $argv[3], $charset);
        break;
    case "NTLM":
        generateNTLMTestHashes($argv[2], $argv[3], $charset);
        break;
    case "SHA1":
        generateSHA1TestHashes($argv[2], $argv[3], $charset);
        break;
    case "SHA256":
        generateSHA256TestHashes($argv[2], $argv[3], $charset);
        break;
    case "SHA":
        generateSHATestHashes($argv[2], $argv[3], $charset);
        break;
    case "MD5_PS":
        generateMD5_PSTestHashes($argv[2], $argv[3], $charset, $argv[4]);
        break;
    case "IPB":
        generateIPBTestHashes($argv[2], $argv[3], $charset);
        break;
    case "PHPASS":
        generatePhpassTestHashes($argv[2], $argv[3], $charset);
        break;
    default:
        print "Unknown hash type {$argv[1]}\n";
        exit;
}

function generatePhpassTestHashes($numPasswords, $passLength, $charset) {
    for ($password = 0; $password < $numPasswords; $password++) {
        $passString = "";
        for ($length = 0; $length < $passLength; $length++) {
            $passString .= $charset[rand(0, strlen($charset) - 1)];
        }
        print phpbb_hash($passString) . "\n";
    }
}


function generateMD5TestHashes($numPasswords, $passLength, $charset) {
    for ($password = 0; $password < $numPasswords; $password++) {
        $passString = "";
        for ($length = 0; $length < $passLength; $length++) {
            $passString .= $charset[rand(0, strlen($charset) - 1)];
        }
        print md5($passString) . "\n";
    }
}

function generateMD5_PSTestHashes($numPasswords, $passLength, $charset, $maxSaltLength) {
    for ($password = 0; $password < $numPasswords; $password++) {
        $passString = "";
        $saltString = "";
        $saltLength = rand(1, $maxSaltLength);
        for ($length = 0; $length < $saltLength; $length++) {
            $saltString .= $charset[rand(0, strlen($charset) - 1)];
        }
        for ($length = 0; $length < $passLength; $length++) {
            $passString .= $charset[rand(0, strlen($charset) - 1)];
        }
        print bin2hex($saltString) . ":" . md5($passString.$saltString) . "\n";
    }
}

// IPB hashes: md5(md5($salt).md5($pass)) - salt length 5.
function generateIPBTestHashes($numPasswords, $passLength, $charset) {
    global $ipbSaltCharset;
    for ($password = 0; $password < $numPasswords; $password++) {
        $passString = "";
        $saltString = "";
        $saltLength = 5;
        for ($length = 0; $length < $saltLength; $length++) {
            $saltString .= $ipbSaltCharset[rand(0, strlen($ipbSaltCharset) - 1)];
        }
        for ($length = 0; $length < $passLength; $length++) {
            $passString .= $charset[rand(0, strlen($charset) - 1)];
        }
        print md5(md5($saltString).md5($passString)) . ":" . $saltString . "\n";
    }
}

function generateSHA1TestHashes($numPasswords, $passLength, $charset) {
    for ($password = 0; $password < $numPasswords; $password++) {
        $passString = "";
        for ($length = 0; $length < $passLength; $length++) {
            $passString .= $charset[rand(0, strlen($charset) - 1)];
        }
        print sha1($passString) . "\n";
    }
}

// Generate LDAP {SHA} hashes - base64 encode the SHA1 binary data.
function generateSHATestHashes($numPasswords, $passLength, $charset) {
    for ($password = 0; $password < $numPasswords; $password++) {
        $passString = "";
        for ($length = 0; $length < $passLength; $length++) {
            $passString .= $charset[rand(0, strlen($charset) - 1)];
        }
        // base64 encode the binary output, not the ascii output
        print '{SHA}' . base64_encode(sha1($passString, 1)) . "\n";
    }
}

function generateSHA256TestHashes($numPasswords, $passLength, $charset) {
    for ($password = 0; $password < $numPasswords; $password++) {
        $passString = "";
        for ($length = 0; $length < $passLength; $length++) {
            $passString .= $charset[rand(0, strlen($charset) - 1)];
        }
        print hash("SHA256", $passString) . "\n";
    }
}

function generateNTLMTestHashes($numPasswords, $passLength, $charset) {
    $MD4 = new MD4;
    for ($password = 0; $password < $numPasswords; $password++) {
        $passString = "";
        for ($length = 0; $length < $passLength; $length++) {
            $passString .= $charset[rand(0, strlen($charset) - 1)] . chr(0);
        }
        print strtoupper($MD4->Calc($passString)) . "\n";
    }
}










################################################################################
#                                                                              #
# MD4 pure PHP edition by DKameleon (http://dkameleon.com)                     #
#                                                                              #
# A PHP implementation of the RSA Data Security, Inc. MD4 Message              #
# Digest Algorithm, as defined in RFC 1320.                                    #
# Based on JavaScript realization taken from: http://pajhome.org.uk/crypt/md5/ #
#                                                                              #
# Updates and new versions: http://my-tools.net/md4php/                        #
#                                                                              #
# History of changes:                                                          #
# 2007.04.06                                                                   #
# - initial release                                                            #
# 2007.04.15                                                                   #
# - fixed safe_add function                                                    #
# 2007.08.26                                                                   #
# - changed code to single class implementation                                #
# - changed safe_add function a little                                         #
# - added self test function                                                   #
# 2009.01.16                                                                   #
# - added some optimizations suggested (by Alex Polushin)                      #
#                                                                              #
################################################################################

# MD4 class
class MD4 {

        var $mode = 0; // safe_add mode. got one report about optimization


        function MD4($selftest = true) {
                if ($selftest) { $this->SelfTest(); }
        }


        function SelfTest() {
                $result = $this->Calc("12345678") == "012d73e0fab8d26e0f4d65e36077511e";
                $this->mode = $result ? 0 : 1;
                return $result;
        }


        function str2blks($str) {
                $nblk = ((strlen($str) + 8) >> 6) + 1;
                for($i = 0; $i < $nblk * 16; $i++) $blks[$i] = 0;
                for($i = 0; $i < strlen($str); $i++)
                        $blks[$i >> 2] |= ord($str{$i}) << (($i % 4) * 8);
                $blks[$i >> 2] |= 0x80 << (($i % 4) * 8);
                $blks[$nblk * 16 - 2] = strlen($str) * 8;
                return $blks;
        }


        function safe_add($x, $y) {
                if ($this->mode == 0) {
                        return ($x + $y) & 0xFFFFFFFF;
                }

                $lsw = ($x & 0xFFFF) + ($y & 0xFFFF);
                $msw = ($x >> 16) + ($y >> 16) + ($lsw >> 16);
                return ($msw << 16) | ($lsw & 0xFFFF);
        }


        function zeroFill($a, $b) {
                $z = hexdec(80000000);
                if ($z & $a) {
                        $a >>= 1;
                        $a &= (~$z);
                        $a |= 0x40000000;
                        $a >>= ($b-1);
                } else {
                        $a >>= $b;
                }
                return $a;
        }


        function rol($num, $cnt) {
                return ($num << $cnt) | ($this->zeroFill($num, (32 - $cnt)));
        }


        function cmn($q, $a, $b, $x, $s, $t) {
                return $this->safe_add($this->rol($this->safe_add($this->safe_add($a, $q), $this->safe_add($x, $t)), $s), $b);
        }


        function ffMD4($a, $b, $c, $d, $x, $s) {
                return $this->cmn(($b & $c) | ((~$b) & $d), $a, 0, $x, $s, 0);
        }


        function ggMD4($a, $b, $c, $d, $x, $s) {
                return $this->cmn(($b & $c) | ($b & $d) | ($c & $d), $a, 0, $x, $s, 1518500249);
        }


        function hhMD4($a, $b, $c, $d, $x, $s) {
                return $this->cmn($b ^ $c ^ $d, $a, 0, $x, $s, 1859775393);
        }


        function Calc($str, $raw_output = false) {

                $x = $this->str2blks($str);

                $a =  1732584193;
                $b = -271733879;
                $c = -1732584194;
                $d =  271733878;

                for($i = 0; $i < count($x); $i += 16) {
                        $olda = $a;
                        $oldb = $b;
                        $oldc = $c;
                        $oldd = $d;

                        $a = $this->ffMD4($a, $b, $c, $d, $x[$i+ 0], 3 );
                        $d = $this->ffMD4($d, $a, $b, $c, $x[$i+ 1], 7 );
                        $c = $this->ffMD4($c, $d, $a, $b, $x[$i+ 2], 11);
                        $b = $this->ffMD4($b, $c, $d, $a, $x[$i+ 3], 19);
                        $a = $this->ffMD4($a, $b, $c, $d, $x[$i+ 4], 3 );
                        $d = $this->ffMD4($d, $a, $b, $c, $x[$i+ 5], 7 );
                        $c = $this->ffMD4($c, $d, $a, $b, $x[$i+ 6], 11);
                        $b = $this->ffMD4($b, $c, $d, $a, $x[$i+ 7], 19);
                        $a = $this->ffMD4($a, $b, $c, $d, $x[$i+ 8], 3 );
                        $d = $this->ffMD4($d, $a, $b, $c, $x[$i+ 9], 7 );
                        $c = $this->ffMD4($c, $d, $a, $b, $x[$i+10], 11);
                        $b = $this->ffMD4($b, $c, $d, $a, $x[$i+11], 19);
                        $a = $this->ffMD4($a, $b, $c, $d, $x[$i+12], 3 );
                        $d = $this->ffMD4($d, $a, $b, $c, $x[$i+13], 7 );
                        $c = $this->ffMD4($c, $d, $a, $b, $x[$i+14], 11);
                        $b = $this->ffMD4($b, $c, $d, $a, $x[$i+15], 19);

                        $a = $this->ggMD4($a, $b, $c, $d, $x[$i+ 0], 3 );
                        $d = $this->ggMD4($d, $a, $b, $c, $x[$i+ 4], 5 );
                        $c = $this->ggMD4($c, $d, $a, $b, $x[$i+ 8], 9 );
                        $b = $this->ggMD4($b, $c, $d, $a, $x[$i+12], 13);
                        $a = $this->ggMD4($a, $b, $c, $d, $x[$i+ 1], 3 );
                        $d = $this->ggMD4($d, $a, $b, $c, $x[$i+ 5], 5 );
                        $c = $this->ggMD4($c, $d, $a, $b, $x[$i+ 9], 9 );
                        $b = $this->ggMD4($b, $c, $d, $a, $x[$i+13], 13);
                        $a = $this->ggMD4($a, $b, $c, $d, $x[$i+ 2], 3 );
                        $d = $this->ggMD4($d, $a, $b, $c, $x[$i+ 6], 5 );
                        $c = $this->ggMD4($c, $d, $a, $b, $x[$i+10], 9 );
                        $b = $this->ggMD4($b, $c, $d, $a, $x[$i+14], 13);
                        $a = $this->ggMD4($a, $b, $c, $d, $x[$i+ 3], 3 );
                        $d = $this->ggMD4($d, $a, $b, $c, $x[$i+ 7], 5 );
                        $c = $this->ggMD4($c, $d, $a, $b, $x[$i+11], 9 );
                        $b = $this->ggMD4($b, $c, $d, $a, $x[$i+15], 13);

                        $a = $this->hhMD4($a, $b, $c, $d, $x[$i+ 0], 3 );
                        $d = $this->hhMD4($d, $a, $b, $c, $x[$i+ 8], 9 );
                        $c = $this->hhMD4($c, $d, $a, $b, $x[$i+ 4], 11);
                        $b = $this->hhMD4($b, $c, $d, $a, $x[$i+12], 15);
                        $a = $this->hhMD4($a, $b, $c, $d, $x[$i+ 2], 3 );
                        $d = $this->hhMD4($d, $a, $b, $c, $x[$i+10], 9 );
                        $c = $this->hhMD4($c, $d, $a, $b, $x[$i+ 6], 11);
                        $b = $this->hhMD4($b, $c, $d, $a, $x[$i+14], 15);
                        $a = $this->hhMD4($a, $b, $c, $d, $x[$i+ 1], 3 );
                        $d = $this->hhMD4($d, $a, $b, $c, $x[$i+ 9], 9 );
                        $c = $this->hhMD4($c, $d, $a, $b, $x[$i+ 5], 11);
                        $b = $this->hhMD4($b, $c, $d, $a, $x[$i+13], 15);
                        $a = $this->hhMD4($a, $b, $c, $d, $x[$i+ 3], 3 );
                        $d = $this->hhMD4($d, $a, $b, $c, $x[$i+11], 9 );
                        $c = $this->hhMD4($c, $d, $a, $b, $x[$i+ 7], 11);
                        $b = $this->hhMD4($b, $c, $d, $a, $x[$i+15], 15);

                        $a = $this->safe_add($a, $olda);
                        $b = $this->safe_add($b, $oldb);
                        $c = $this->safe_add($c, $oldc);
                        $d = $this->safe_add($d, $oldd);
                }
                $x = pack('V4', $a, $b, $c, $d);
                if ($raw_output) { return $x; }
                return bin2hex($x);
        }


}
# MD4 class

/**
*
* @version Version 0.1 / $Id: functions.php 8491 2008-04-04 11:41:58Z acydburn $
*
* Portable PHP password hashing framework.
*
* Written by Solar Designer <solar at openwall.com> in 2004-2006 and placed in
* the public domain.
*
* There's absolutely no warranty.
*
* The homepage URL for this framework is:
*
*	http://www.openwall.com/phpass/
*
* Please be sure to update the Version line if you edit this file in any way.
* It is suggested that you leave the main version number intact, but indicate
* your project name (after the slash) and add your own revision information.
*
* Please do not change the "private" password hashing method implemented in
* here, thereby making your hashes incompatible.  However, if you must, please
* change the hash type identifier (the "$P$") to something different.
*
* Obviously, since this code is in the public domain, the above are not
* requirements (there can be none), but merely suggestions.
*
*
* Hash the password
*/
function phpbb_hash($password)
{
	$itoa64 = './0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';

	$random_state = uniqid();
	$random = '';
	$count = 6;

	if (($fh = @fopen('/dev/urandom', 'rb')))
	{
		$random = fread($fh, $count);
		fclose($fh);
	}

	if (strlen($random) < $count)
	{
		$random = '';

		for ($i = 0; $i < $count; $i += 16)
		{
			$random_state = md5(unique_id() . $random_state);
			$random .= pack('H*', md5($random_state));
		}
		$random = substr($random, 0, $count);
	}

	$hash = _hash_crypt_private($password, _hash_gensalt_private($random, $itoa64), $itoa64);

	if (strlen($hash) == 34)
	{
		return $hash;
	}

	return md5($password);
}

/**
* Check for correct password
*/
function phpbb_check_hash($password, $hash)
{
	$itoa64 = './0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
	if (strlen($hash) == 34)
	{
		return (_hash_crypt_private($password, $hash, $itoa64) === $hash) ? true : false;
	}

	return (md5($password) === $hash) ? true : false;
}

/**
* Generate salt for hash generation
*/
function _hash_gensalt_private($input, &$itoa64, $iteration_count_log2 = 6)
{
	if ($iteration_count_log2 < 4 || $iteration_count_log2 > 31)
	{
		$iteration_count_log2 = 8;
	}

	$output = '$H$';
	$output .= $itoa64[min($iteration_count_log2 + ((PHP_VERSION >= 5) ? 5 : 3), 30)];
	$output .= _hash_encode64($input, 6, $itoa64);

	return $output;
}

/**
* Encode hash
*/
function _hash_encode64($input, $count, &$itoa64)
{
	$output = '';
	$i = 0;
        
	do
	{
		$value = ord($input[$i++]);
		$output .= $itoa64[$value & 0x3f];

		if ($i < $count)
		{
			$value |= ord($input[$i]) << 8;
		}

		$output .= $itoa64[($value >> 6) & 0x3f];

		if ($i++ >= $count)
		{
			break;
		}

		if ($i < $count)
		{
			$value |= ord($input[$i]) << 16;
		}

		$output .= $itoa64[($value >> 12) & 0x3f];

		if ($i++ >= $count)
		{
			break;
		}

		$output .= $itoa64[($value >> 18) & 0x3f];
	}
	while ($i < $count);
        
	return $output;
}

/**
* The crypt function/replacement
*/
function _hash_crypt_private($password, $setting, &$itoa64)
{
    $output = '*';

	// Check for correct hash
	if (substr($setting, 0, 3) != '$H$')
	{
		return $output;
	}

	$count_log2 = strpos($itoa64, $setting[3]);

	if ($count_log2 < 7 || $count_log2 > 30)
	{
		return $output;
	}

	$count = 1 << $count_log2;
	$salt = substr($setting, 4, 8);
        
	if (strlen($salt) != 8)
	{
		return $output;
	}

	/**
	* We're kind of forced to use MD5 here since it's the only
	* cryptographic primitive available in all versions of PHP
	* currently in use.  To implement our own low-level crypto
	* in PHP would result in much worse performance and
	* consequently in lower iteration counts and hashes that are
	* quicker to crack (by non-PHP code).
	*/
	if (PHP_VERSION >= 5)
	{
		$hash = md5($salt . $password, true);
		do
		{
			$hash = md5($hash . $password, true);
		}
		while (--$count);
	}
	else
	{
		$hash = pack('H*', md5($salt . $password));
		do
		{
			$hash = pack('H*', md5($hash . $password));
		}
		while (--$count);
	}
        
	$output = substr($setting, 0, 12);
	$output .= _hash_encode64($hash, 16, $itoa64);

	return $output;
}

// END phpass code

?>
