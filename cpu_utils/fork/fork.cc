#include <stdio.h> 
#include <sys/types.h> 
#include <unistd.h> 


int main() { 
  fork(); 
  printf("hello\n");
  return 0; 
} 
