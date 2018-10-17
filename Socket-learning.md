# Learning of the Socket 
Briefly speaking, Socket is like a necessary file IO for distributed computer systems. The client will first send requests to establish a connection to a specific server. If the connection is accepted, it will continuously send out requests through Socket and execute in and out operations, this is the foundation of most web services.   

The  three key points for start a socket service is shown below:  
  + Socket Table
  + Socket ADDR_In 
      1. Address Family
      2. IP Address
      3. Port Number 
  + Host Entry  
 
 Where the socket table means the group of socket operations, mostly are the input-output format. The Socket Address system has two important parameters.  In client side, the client should define the IP address to send the request with which port in that address this request will use. In server side, the server should define which IP addresses have permission to reach this services. The host entry means that a domain name service (DNS) which could resolve domain name and return a IP address. 

