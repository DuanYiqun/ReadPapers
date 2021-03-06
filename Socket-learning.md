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
     if (sockfd < 0) 
        error("ERROR opening socket");
     bzero((char *) &serv_addr, sizeof(serv_addr));
```
Here define sockfd want a stream connection, which means it uses TCP and connected oriented. 0 means wipe out the address in ram to prevent ram error. 

```C
     portno = atoi(argv[1]);
```
argv[0] is the name of the program. argv[1] means the first string that passed in. 
atoi() transfer ASCII to integer. 
This code give portnumber to main entry. The ports below 1024 are previllage ports in most of the programing languages. The common port number including 5000 9000 8000 in python. 


```C
     serv_addr.sin_family = AF_INET;
     serv_addr.sin_addr.s_addr = INADDR_ANY;
     serv_addr.sin_port = htons(portno);
```
.sin means socket internet. 
htons means host byte order changed to network byte order

```C
     if (bind(sockfd, (struct sockaddr *) &serv_addr,
              sizeof(serv_addr)) < 0) 
              error("ERROR on binding");
```
this operation actually fill in socket fd with ip address and port number. If the content is blank, return error.

listen(sockfd,5);
     clilen = sizeof(cli_addr);
```C
     listen(sockfd,5);
     clilen = sizeof(cli_addr);
     newsockfd = accept(sockfd, 
                 (struct sockaddr *) &cli_addr, 
                 &clilen);
     if (newsockfd < 0) 
          error("ERROR on accept");
                 
```
This operation means that the sockfd are always listening to a specific port then give it to newsockfd. And tell it who is connected. If newsock haven't receive any connections then raise error. 

```C
    bzero(buffer,256);
    n = read(newsockfd,buffer,255);
    if (n < 0) error("ERROR reading from socket");
    printf("Here is the message: %s\n",buffer);         
```
Firstly read buffer from socket, n tells how much bits receives. 255 in the function restrain the read function from over writing the buffer.   
If n <0, which means there is actually no bits readed, we raise error. 


```C
    n = write(newsockfd,"I got your message",18);
    if (n < 0) error("ERROR writing to socket");  
    close(newsockfd);
    close(sockfd);
    return 0; 
```

## Client.c
The same as server.c so we move faster this time. 

```C
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 

void error(const char *msg)
{
    perror(msg);
    exit(0);
}

int main(int argc, char *argv[])
{
    int sockfd, portno, n;
    struct sockaddr_in serv_addr;
    struct hostent *server;

    char buffer[256];
    if (argc < 3) {
       fprintf(stderr,"usage %s hostname port\n", argv[0]);
       exit(0);
    }
    portno = atoi(argv[2]);
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) 
        error("ERROR opening socket");
    server = gethostbyname(argv[1]);
    if (server == NULL) {
        fprintf(stderr,"ERROR, no such host\n");
        exit(0);
    }
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr, 
         (char *)&serv_addr.sin_addr.s_addr,
         server->h_length);
    serv_addr.sin_port = htons(portno);
    if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0) 
        error("ERROR connecting");
    printf("Please enter the message: ");
    bzero(buffer,256);
    fgets(buffer,255,stdin);
    n = write(sockfd,buffer,strlen(buffer));
    if (n < 0) 
         error("ERROR writing to socket");
    bzero(buffer,256);
    n = read(sockfd,buffer,255);
    if (n < 0) 
         error("ERROR reading from socket");
    printf("%s\n",buffer);
    close(sockfd);
    return 0;
}
```
The only difference server = gethostbyname(argv[1]);   
Actually we send domain name and gethostbyname() return the IP.   
And client firstly write then read, the server firstly read then write.

## 总结
用中文总结一下，socket 主要有四个关键参数，第一个是socketfd即file descriptor。 第二和第三个是socket table和 一系列IP address，port number， 第四个是DNS解析，即给定域名，返回一个IP地址。 主要的流程为不断地尝试操作，如果不可行则抛出错误的过程。 在连接方面使用TCP连接。在操作时要特别注意buffer溢出的问题。在client端，系统是先写后读，在server端是先读后写。



