# The 7 layers of OSI model 
Here we directly give the 7 layers of a OSI model below:

  + Layer 7 - Application : Email/webservices.
  + Layer 6 - Presentation
  + Layer 5 - Session
  + Layer 4 - Transport
  + Layer 3 - Network
  + Layer 2 - Data Link
  + Layer 1 - Physical

This page are mostly talk about the transmission layer.   
Concepts:  
  + _Header_ : Info attached to the head of data.
  + _Segment_ : Sliced pieces of Datagram. 
Acknowledged : All network is unreliable, so we need to make best effort to make data transfer secure.   

Transmission layer should promise:
 + FLow control : Prevent overwhelming between two devices happens. 
 + Conjestion control : Prevent one powerful device take others' brandwidth thus make conjestion 
 + Able to transfer data under unstable network environment.
 
Multiplexing:
Multiple signal share one channel.    
Demultiplexing:
Decode signal from one channel to several signals.   
In productive environments, they use port to represent this process.   
_Ports_ : 
+ Unique identifier
+ Has 16 bits so totally the range of the port number is: 0- 2^16
+ 0-1023 are reserved ports, so we need to use other ports.
When a socket process is opened, it will be randomly assigne to a specific port.

