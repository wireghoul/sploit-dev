<?php	
	// LITTLE ENDIAN ONLY!
	
	$numericData = explode("\n", file_get_contents("Passwords-Numeric.txt"));
	
	makeHashFiles("Numeric", $numericData);
	
	$textData = explode("\n", file_get_contents("Passwords-Full.txt"));
	
	makeHashFiles("Full", $textData);	
	
	
	
	function makeHashFiles($class, $data) {
		$MD4 = new MD4;
		
		$NTLMOUT = "";
		$MD5OUT = "";
		$DMD5OUT = "";
		$MD4OUT = "";
		$SHA1OUT = "";
		$DUPMD5OUT = "";
		$DUPNTLMOUT = "";
                $LMOUT = "";
                $SHA256OUT = "";

		for ($i = 0; $i < count($data); $i++) {
			if (strlen($data[$i]) == 0) {
				continue;
			}
			$MD5OUT .= strtoupper(md5($data[$i])) . "\n";
			$DMD5OUT .= strtoupper(md5(md5($data[$i]))) . "\n";
			$DUPMD5OUT .= strtoupper(md5($data[$i] . $data[$i])) . "\n";
			$MD4OUT .= strtoupper($MD4->Calc($data[$i])) . "\n";
			$SHA1OUT .= strtoupper(sha1($data[$i])) . "\n";
			$ntlmString = "";
			for ($j = 0; $j < strlen($data[$i]); $j++) {
				$ntlmString .= $data[$i][$j] . chr(0);
			}
			$NTLMOUT .= strtoupper($MD4->Calc($ntlmString)) . "\n";
                        $DUPNTLMOUT .= strtoupper($MD4->Calc($ntlmString.$ntlmString)) . "\n";
                        $LMOUT .= LMhash($data[$i]) . "\n";
                        $SHA256OUT .= hash('sha256', $data[$i]) . "\n";
			
		}
		file_put_contents("Hashes-MD4-$class.txt",$MD4OUT);
		file_put_contents("Hashes-MD5-$class.txt",$MD5OUT);
		file_put_contents("Hashes-DMD5-$class.txt",$DMD5OUT);
		file_put_contents("Hashes-DupMD5-$class.txt",$DUPMD5OUT);
		file_put_contents("Hashes-SHA1-$class.txt",$SHA1OUT);
		file_put_contents("Hashes-NTLM-$class.txt",$NTLMOUT);
		file_put_contents("Hashes-DupNTLM-$class.txt",$DUPNTLMOUT);
		file_put_contents("Hashes-LM-$class.txt",$LMOUT);
		file_put_contents("Hashes-SHA256-$class.txt",$SHA256OUT);
		
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
	
function LMhash($string)
{
    $string = strtoupper(substr($string,0,14));

    $p1 = LMhash_DESencrypt(substr($string, 0, 7));
    $p2 = LMhash_DESencrypt(substr($string, 7, 7));

    return strtoupper($p1.$p2);
}

function LMhash_DESencrypt($string)
{
    $key = array();
    $tmp = array();
    $len = strlen($string);

    for ($i=0; $i<7; ++$i)
        $tmp[] = $i < $len ? ord($string[$i]) : 0;

    $key[] = $tmp[0] & 254;
    $key[] = ($tmp[0] << 7) | ($tmp[1] >> 1);
    $key[] = ($tmp[1] << 6) | ($tmp[2] >> 2);
    $key[] = ($tmp[2] << 5) | ($tmp[3] >> 3);
    $key[] = ($tmp[3] << 4) | ($tmp[4] >> 4);
    $key[] = ($tmp[4] << 3) | ($tmp[5] >> 5);
    $key[] = ($tmp[5] << 2) | ($tmp[6] >> 6);
    $key[] = $tmp[6] << 1;

    $is = mcrypt_get_iv_size(MCRYPT_DES, MCRYPT_MODE_ECB);
    $iv = mcrypt_create_iv($is, MCRYPT_RAND);
    $key0 = "";

    foreach ($key as $k)
        $key0 .= chr($k);
    $crypt = mcrypt_encrypt(MCRYPT_DES, $key0, "KGS!@#$%", MCRYPT_MODE_ECB, $iv);

    return bin2hex($crypt);
}

	
	?>
