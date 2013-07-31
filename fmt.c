#include <stdio.h>

int main(int argc, char *argv[])
{

char buf[1024+1];

if(argc < 2) { printf("usage: %s data\n", argv[0]); return 0; }

     snprintf(buf, sizeof(buf)-1, "%s", argv[1]);
     printf(buf); // clearly a format string bug
     printf("\n");

return 0;
}
