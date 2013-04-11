#!/usr/bin/perl
@badchars=(hex('00'),hex('0a'),hex('0d'),hex('20'));
for ($x = 255; $x>1; $x-=4) {
  for ($y = $x-4; $y < $x;$y++) {
    # Filter characters here
    printf "\\x%02d",$y if (! grep /^$y$/, @badchars and $y >= 0);
  }
}
