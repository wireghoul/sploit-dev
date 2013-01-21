<?php

/*
            switch (PASSWORD_LENGTH) {
                case 8:
                    MFNSingleIncrementorsOpenCL8 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 0);
                    MFNSingleIncrementorsOpenCL8 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 1);
                    MFNSingleIncrementorsOpenCL8 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 2);
                    MFNSingleIncrementorsOpenCL8 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 3);
                break;

                case 7:
                    MFNSingleIncrementorsOpenCL7 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 0);
                    MFNSingleIncrementorsOpenCL7 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 1);
                    MFNSingleIncrementorsOpenCL7 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 2);
                    MFNSingleIncrementorsOpenCL7 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 3);
                break;
                
                case 6:
                    MFNSingleIncrementorsOpenCL6 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 0);
                    MFNSingleIncrementorsOpenCL6 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 1);
                    MFNSingleIncrementorsOpenCL6 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 2);
                    MFNSingleIncrementorsOpenCL6 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 3);
                break;
                
                case 5:
                    MFNSingleIncrementorsOpenCL5 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 0);
                    MFNSingleIncrementorsOpenCL5 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 1);
                    MFNSingleIncrementorsOpenCL5 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 2);
                    MFNSingleIncrementorsOpenCL5 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 3);
                break;
                
                case 4:
                    MFNSingleIncrementorsOpenCL4 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 0);
                    MFNSingleIncrementorsOpenCL4 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 1);
                    MFNSingleIncrementorsOpenCL4 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 2);
                    MFNSingleIncrementorsOpenCL4 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 3);
                break;

                case 3:
                    MFNSingleIncrementorsOpenCL3 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 0);
                    MFNSingleIncrementorsOpenCL3 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 1);
                    MFNSingleIncrementorsOpenCL3 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 2);
                    MFNSingleIncrementorsOpenCL3 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 3);
                break;
                
                case 2:
                    MFNSingleIncrementorsOpenCL2 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 0);
                    MFNSingleIncrementorsOpenCL2 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 1);
                    MFNSingleIncrementorsOpenCL2 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 2);
                    MFNSingleIncrementorsOpenCL2 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 3);
                break;
                
                case 1:
                    MFNSingleIncrementorsOpenCL1 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 0);
                    MFNSingleIncrementorsOpenCL1 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 1);
                    MFNSingleIncrementorsOpenCL1 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 2);
                    MFNSingleIncrementorsOpenCL1 (sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5, 3);
                break;

            }
*/

// Max password length to generate for.
$maxPassLen = 16;
$incrementorBase = "MFNSingleIncrementorsOpenCL";
// Call parameters to use to the incrementor call.  Final comma needed.
$callParamsFirst = "sharedCharsetPlainMD5, sharedReverseCharsetPlainMD5, sharedCharsetLengthsPlainMD5,";

print "            switch (PASSWORD_LENGTH) {\n";

for ($i = 1; $i <= $maxPassLen; $i++) {
    print "                case $i:\n";
    print "#if grt_vector_1 || grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16\n";
    print "                    $incrementorBase{$i} ($callParamsFirst 0);\n";
    print "#endif\n";
    print "#if grt_vector_2 || grt_vector_4 || grt_vector_8 || grt_vector_16\n";
    print "                    $incrementorBase{$i} ($callParamsFirst 1);\n";
    print "#endif\n";
    print "#if grt_vector_4 || grt_vector_8 || grt_vector_16\n";
    print "                    $incrementorBase{$i} ($callParamsFirst 2);\n";
    print "                    $incrementorBase{$i} ($callParamsFirst 3);\n";
    print "#endif\n";
    print "#if grt_vector_8 || grt_vector_16\n";
    print "                    $incrementorBase{$i} ($callParamsFirst 4);\n";
    print "                    $incrementorBase{$i} ($callParamsFirst 5);\n";
    print "                    $incrementorBase{$i} ($callParamsFirst 6);\n";
    print "                    $incrementorBase{$i} ($callParamsFirst 7);\n";
    print "#endif\n";
    print "#if grt_vector_16\n";
    print "                    $incrementorBase{$i} ($callParamsFirst 8);\n";
    print "                    $incrementorBase{$i} ($callParamsFirst 9);\n";
    print "                    $incrementorBase{$i} ($callParamsFirst 10);\n";
    print "                    $incrementorBase{$i} ($callParamsFirst 11);\n";
    print "                    $incrementorBase{$i} ($callParamsFirst 12);\n";
    print "                    $incrementorBase{$i} ($callParamsFirst 13);\n";
    print "                    $incrementorBase{$i} ($callParamsFirst 14);\n";
    print "                    $incrementorBase{$i} ($callParamsFirst 15);\n";
    print "#endif\n";
    print "                break;\n";
}
    print "            }\n";
