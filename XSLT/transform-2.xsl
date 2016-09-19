 <xsl:stylesheet version="1.0"
   xmlns:atom="http://www.w3.org/2005/Atom"
   xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
 
     <!-- The "feed" tag -->
     <xsl:template match="/atom:feed">
         <h1><xsl:value-of select="atom:title"/><br/></h1>
         <xsl:apply-templates select="atom:entry"/>
     </xsl:template>
 
     <!-- The "entry" tags -->
     <xsl:template match="atom:entry">
         <hr/>Entry #<xsl:value-of select="position()"/>
         [<xsl:value-of select="string-length(atom:title)"/>] :
         <pre><xsl:value-of select="atom:title"/></pre>
     </xsl:template>
 
 </xsl:stylesheet>
