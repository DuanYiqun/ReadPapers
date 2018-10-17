# Learning of the Socket 
If possible will add TCP/IP basic knowledges here.

Briefly speaking, Socket is like a necessary file IO for distributed computer systems. The client will first send requests to establish a connection to a specific server. If the connection is accepted, it will continuously send out requests through Socket and execute in and out operations, this is the foundation of most web services.   

The  three key points for start a socket service is shown below:  
  + Socket FD
  + Socket Table
  + Socket ADDR_In 
      1. Address Family
      2. IP Address
      3. Port Number 
  + Host Entry  
 
Where Socket FD means the Socket file descriptor for writing and reading file. Here, file descriptors are part of the POSIX application programming interface. A file descriptor is a non-negative integer, represented in C programming language as the type int. The socket table means the group of socket operations, mostly are the input-output format. The Socket Address system has two important parameters.  In client side, the client should define the IP address to send the request with which port in that address this request will use. In server side, the server should define which IP addresses have permission to reach this services. The host entry means that a domain name service (DNS) which could resolve domain name and return a IP address. 

Then we directly go to the source code in C:
# server.c
```C
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
```
This file head package declaration define basic dependency for socket.
```C
void error(const char *msg)
{
    perror(msg);
    exit(1);
}
```
This function is a error processing function, return a declared error message then exit. Actually the whole file is keep doing the operation like Python try:.. except: .. operation. This is because network structure might have lots of connection problems, if we raise error, we could understand which part is down. 

```C
int main(int argc, char *argv[])
{
     int sockfd, newsockfd, portno;
     socklen_t clilen;
     char buffer[256];
     struct sockaddr_in serv_addr, cli_addr;
     int n;
     if (argc < 2) {
         fprintf(stderr,"ERROR, no port provided\n");
         exit(1);
     }
```

The int socktfd, newsocketfd means the file descriptor. portno means the port number. 
Char buffer is actually where to write and read data, limited under 256 bits.   
If buffer are writed over 256, mostly will got buffer overrun error.   
At last if there is no port number provided raise error. 

```C
     sockfd = socket(AF_INET, SOCK_STREAM, 0);
```


