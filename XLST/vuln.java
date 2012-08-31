 <%@ page language="java" contentType="text/html" %>
 <%@ page import="javax.xml.transform.*"%>
 <%@ page import="javax.xml.transform.stream.*"%>
 <%
 
 // Echo some info
 out.print("This ATOM reader is coded in JSP+XSLT, using Tomcat and Xalan-J<br/>");
 
 // Get the parameters
 String xmlFile    = request.getParameter("url");
 String xslFile    = request.getParameter("xsl");
 
 // Create a XSLT transformer
 TransformerFactory tFactory = TransformerFactory.newInstance();
 
 // Configure the XSLT stylesheet
 Transformer transformer = tFactory.newTransformer(new StreamSource(xslFile));
 
 // Transform the XML file
 transformer.transform(new StreamSource(xmlFile), new StreamResult(out));
 %>
