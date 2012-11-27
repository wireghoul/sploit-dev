#!/usr/bin/perl -w

# FUZZMAN
#=========================
# man page based fuzzing |
# by Emmanouel Kellinis  |
# me at cipher org uk    |
# Extended by Wireghoul  |
#=========================



#print @ARGV;

$num_args = @ARGV;

if ($num_args != 1)
{
        print "\nFuzzMan ERROR:\nYou need to supply the command you want to fuzz, e.g. perl fuzzman.pl date\n\n\n";
        exit 1;
}


my $manpage = $ARGV[0];
my $countargs = 0;


system "man $manpage  > $manpage.man / 2>/dev/null" ;

#Signal Handling
$SIG{INT}  = 'sig';


$MANFILE = "$manpage.man";
open(MANFILE) or die("Could not open man page");

print "\n=== Extract arguments for \"$manpage\" ===\n"; 

#SEND to Extractor
my @catchem; 
my @argsarray = ();

print "\tSTANDARD\n";

foreach $line (<MANFILE>) {
@catchem = reg($line);
}
print "\tADDITIONAL\n"; 
	push(@catchem,attack(0));
        print "\t : EXTRA BoF Arg\n";
        push(@catchem,attack(1));
        print "\t : EXTRA Format String Arg\n";
        push(@catchem,attack(2));
        print "\t : EXTRA Numbers  Arg\n";

my (@list) = @catchem;
my %Seen = ();
my @unique = grep { ! $Seen{ $_ }++ } @catchem;

print "\n";
print ":Number of Arguments :".@unique; 
#$countargs";
print "\n\n";
 
my  $fuzz = "";

print "=== Generate Fuzzing Script ===\n";
print "+STOP GENERATOR WITH CTRL-C\n";
$argcom =0;
system "echo \"\" > ".$manpage.".sh  2>/dev/null";

#$out = gen(\@unique);
open (SHFILE, '>'.$manpage.'.sh');
#close(SHFILE);
gen(\@unique);
close (SHFILE);
print "\n:".$manpage.".sh script ready\n";
print ":Run fuzzing script [sh ".$manpage.".sh]\n\n";

#Termination Signal Handling
sub sig {
close (SHFILE);
die "\n:Partial ".$manpage.".sh, not all combinations have been generated\n:Run fuzzing script [sh ".$manpage.".sh]\n";
}


# Generate Input 
sub gen {
   	my ($list) = @_;
   	my (@print, $str, $i, $j);
   	my $size = @{$list};

   	for ($i = 0; $i < 2**$size; $i++) {
      		$str = sprintf("%*.*b", $size, $size, $i);
      		@print = ();
      		for ($j = 0; $j < $size; $j++) {
         	if (substr($str, $j, 1)) { push (@print, $list->[$j]); }
      	}
        print SHFILE "$manpage ".join(' ', @print)." >/dev/null 2>/dev/null &\n";
        print SHFILE "if [ \$? == 139 ]; then echo '$manpage ".join(' ', @print) ."'; fi\n";
	$argcom++;
  print SHFILE "wait \$!\n" if ($argcom % 100 == 0);
	print "\r:Agrument combinations\t: $argcom" if ($argcom % 1000 == 0);
	flush(STDOUT);
        #flush(SHFILE);
   }


#Add crazy number of arguments at the end 
$fuzz = $fuzz."$manpage ".attack(3)."\n";

return $fuzz;
}

# Attack categories
sub attack {
	my ($att) = $_[0];
	#BoF
	my $BoF = "A"x"5001";
	#Format String
	my $FS  = "%s%x%d"x"5";
	#Numbers IoF
	my $IoF = 10000000;
        #Many Arguments
        my $crazyargs = " a"x"200";

	if ($att==0) { return $BoF;}
	elsif ($att==1) { return $FS;} 
	elsif ($att==2) { return $IoF;}
	elsif ($att==3) { return $crazyargs;}
}


# Function = RegExp - Arguments
sub reg {
        my ($arg) = $_[0];
	#print $arg;
	if ($arg=~ /\,\s(\-\-[a-zA-Z-=0-9]{1,30})/) {
        my $option = "$1";
	print "\t : ".$option."\n";
        push(@argsarray,$option);

        if ($option=~ m/\-\-[-a-zA-Z0-9]{1,30}\=/) {
        @left = split(/=/, $option); 
        #Fuzz Options that take argument	
	push(@argsarray,$left[0]."=".attack(0));
	push(@argsarray,$left[0]."=".attack(1));	
	push(@argsarray,$left[0]."=".attack(2));	
	}

        $countargs++;
	}
	elsif ($arg=~ /\s\s\s\s\s\s\s(\-\-[a-zA-Z-=0-9]{1,30})/) {
        my $option = "$1";
        print "\t : ".$option."\n";
        push(@argsarray,$option);

        if ($option=~ m/\-\-[-a-zA-Z0-9]{1,30}\=/) {
        @left = split(/=/, $option);
        #Fuzz Options that take argument
        push(@argsarray,$left[0]."=".attack(0));
        push(@argsarray,$left[0]."=".attack(1));
        push(@argsarray,$left[0]."=".attack(2));
        }
	$countargs++;
        }


        elsif ($arg=~ /(\s\s\-[a-zA-Z]{1,60}),\s/) {
 	my $options = "$1";
        print "\t : ".$options."\n";
        push(@argsarray,$options);
        $countargs++;
	}
	elsif ($arg=~ /(\s\s\s\s\s\s\s\-[a-zA-Z]{1,60})\s/) {
        my $options = "$1";
        print "\t : ".$options."\n";
        push(@argsarray,$options);
        $countargs++;
        }
return @argsarray;
}

sub flush {
   my $h = select($_[0]); my $a=$|; $|=1; $|=$a; select($h);
}
