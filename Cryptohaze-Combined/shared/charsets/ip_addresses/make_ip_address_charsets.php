<?php

/** 
 * Make the files needed for cracking all possible IPv4 address formats.
 * 
 * This implements a quick charset builder for cracking IP address hash formats.
 * 
 * This assumes a flexible IPv4 address format - 192.168.1.1 is valid, as is
 * 192.168.001.001 - we don't discriminate here.
 * 
 * However, if the octet is 3 digits long, the octet will not start with anything
 * but an 0, 1, or 2.  We'll try excess IPs like .299, but this is just a side
 * effect of the charset.  It shouldn't be a big deal.
 */

$bash_script = "#!/bin/bash\n\n";
$bash_script .= "
    if test \"\$4\" == \"\" ; then
    echo Usage: \$0 [path to multiforcer] [path to IP address directory] 
    echo \"    \"[path to IP hash file] [options]
    echo 
    echo [path to multiforcer]: The relative or absolute path to the version of
    echo \"  \"the multiforcer you wish to use.
    echo
    echo [path to IP address directory]: The relative or absolute path to the directory
    echo containing the xxx.xxx.xxx.xxx files - do not include filenames.
    echo
    echo [path to IP hash file]: The relative or absolute path to the file containing
    echo the hashes of interest.
    echo
    echo [options]: All other options to pass to the Multiforcer.  This may be any 
    echo option the Multiforcer supports.  You will definitely need '-h MD5' or 
    echo similar, and will probably want an output file for the found results with
    echo '-o [output file]' - otherwise you will need to scrape the output for the
    echo found hashes.
    echo
    echo
    exit
fi
";

// Possible IP address values for the not-hundreds place
$full_line = "0123456789\n";
// Possible IP address values for the hundreds place
$hundreds_line = "012\n";

$file_count = 1;

for ($o1l = 1; $o1l <= 3; $o1l++) {
    for ($o2l = 1; $o2l <= 3; $o2l++) {
        for ($o3l = 1; $o3l <= 3; $o3l++) {
            for ($o4l = 1; $o4l <= 3; $o4l++) {
                // Create the filename for this option.
                $filename = "";
                $filename .= str_repeat("x", $o1l);
                $filename .= ".";
                $filename .= str_repeat("x", $o2l);
                $filename .= ".";
                $filename .= str_repeat("x", $o3l);
                $filename .= ".";
                $filename .= str_repeat("x", $o4l);
                print "Generating file $filename\n";
                
                $filecontents = "";
                // Switch statements with fallthrough are used.

                switch ($o1l) {
                    case 3:
                        $filecontents .= $hundreds_line;
                    case 2:
                        $filecontents .= $full_line;
                    case 1:
                        $filecontents .= $full_line;
                        $filecontents .= ".\n";
                }

                switch ($o2l) {
                    case 3:
                        $filecontents .= $hundreds_line;
                    case 2:
                        $filecontents .= $full_line;
                    case 1:
                        $filecontents .= $full_line;
                        $filecontents .= ".\n";
                }

                switch ($o3l) {
                    case 3:
                        $filecontents .= $hundreds_line;
                    case 2:
                        $filecontents .= $full_line;
                    case 1:
                        $filecontents .= $full_line;
                        $filecontents .= ".\n";
                }

                switch ($o4l) {
                    case 3:
                        $filecontents .= $hundreds_line;
                    case 2:
                        $filecontents .= $full_line;
                    case 1:
                        $filecontents .= $full_line;
                }
                
                file_put_contents($filename, $filecontents);
                
                // Create the bash script line
                // [script] [path to multiforcer] [path to IP address directory] [path to IP hash file] [options]
                $ip_length = strlen($filename);
                $bash_script .= "$1 -u $2/$filename --min $ip_length --max $ip_length -f $3 $4\n";
                $bash_script .= "echo; echo; echo\n";
                $bash_script .= "echo Progress: $file_count / 81\n";
                $bash_script .= "echo; echo; echo\n";
                $file_count++;
            }
        }
    }
}
file_put_contents("run_ip_brute.sh", $bash_script);