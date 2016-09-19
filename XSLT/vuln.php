 <?php
 
 // Page header
 echo "<html><head>\n";
 echo "<title>ATOM reader (PHP + XSLT)</title>\n";
 echo "<meta http-equiv='Content-Type' content='text/html; charset=UTF-8'/>\n";
 echo "<head><body>\n";
 echo "This ATOM reader is coded in PHP5+XSLT, using libxslt.<br/>\n";
 
 // Get parameters
 $url = $_GET['url'];
 $xsl = $_GET['xsl'];
 
 // Load ATOM file
 $xmldoc = new DOMDocument();
 $xmldoc->load( $url );
 
 // Load XSLT file
 $xsldoc = new DOMDocument();
 $xsldoc->load( $xsl );
 
 // Register PHP functions as XSLT extensions
 $xslt = new XSLTProcessor();
 $xslt->registerPhpFunctions();
 
 // Import the stylesheet
 $xslt->importStylesheet( $xsldoc );
 
 // Transform and print
 print $xslt->transformToXML( $xmldoc );
 
 ?>
