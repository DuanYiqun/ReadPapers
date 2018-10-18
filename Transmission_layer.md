# The 7 layers of OSI model 
Here we directly give the 7 layers of a OSI model below:

  + Layer 7 - Application : Email/web-services.
  + Layer 6 - Presentation
  + Layer 5 - Session
  + Layer 4 - Transport
  + Layer 3 - Network
  + Layer 2 - Data Link
  + Layer 1 - Physical

This page are mostly talk about the transmission layer.   
Concepts:  
  + _Header_ : Info attached to the head of data.
  + _Segment_ : Sliced pieces of Data-gram. 
Acknowledged : All network is unreliable, so we need to make best effort to make data transfer secure.   

Transmission layer should promise:
 + Flow control : Prevent overwhelming between two devices happens. 
 + Congestion control : Prevent one powerful device take others' bandwidth thus make congestion 
 + Able to transfer data under unstable network environment.
 
Multiplexing:
Multiple signal share one channel.    
De-multiplexing:
Decode signal from one channel to several signals.   
In productive environments, they use port to represent this process.   
_Ports_ : 
+ Unique identifier
+ Has 16 bits so totally the range of the port number is: 0- 2^16
+ 0-1023 are reserved ports, so we need to use other ports.
When a socket process is opened, it will be randomly assigned to a specific port.

## UDP : User Data-gram Protocol  
Goal: As simple as possible  
UDP doesn't give any advanced features but only the end to end network. UDP is connection-less protocol, which means
 it do not need to do three-way or four-way handshake for establish a connection. 
 
UDP header:  
Each content is 16 bits wide. It looks like below:       

Source port (optional) |  Destination port 
 Length (Data)       |  checksum (Optional for IPV4) 

