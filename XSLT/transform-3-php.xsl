 <xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:abc="http://php.net/xsl" version="1.0">
         <xsl:template match="/">
                 <xsl:text>Current date: </xsl:text><xsl:value-of select="abc:function('date', 'F j, Y')"/>
         </xsl:template>
 </xsl:stylesheet>
