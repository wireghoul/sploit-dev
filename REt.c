#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv) { 
  char password[] = "yomama";
  if(argc < 2)     {
    printf("Usage: %s <password>\n", argv[0]);
    exit(1);
  }

  if(strncmp(argv[1], password, strlen(password))) {
    printf("FAIL\n");
    exit(1);
  } else {
    printf("WIN\n");
    return(0);
  }

  return(0); // never reached :)
}
