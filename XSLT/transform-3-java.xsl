 <xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:date="http://xml.apache.org/xalan/java/java.util.Date" version="1.0">
         <xsl:template match="/">
                 <xsl:variable name="dateObject" select="date:new()"/>
                 <xsl:text>Current date: </xsl:text><xsl:value-of select="$dateObject"/>
         </xsl:template>
 </xsl:stylesheet>
