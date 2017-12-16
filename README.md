# RSA_CUDA

My second go with CUDA library, this time RSA


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![IDE_CLION](https://img.shields.io/badge/IDE-Visual%20studio-green.svg)](https://www.visualstudio.com/)

## Install and run:
	Using visual Studio solution manager
	
## Requirements!
* CUDA enabled GPU 

## Stage 1
	Builds a random array at the size of random;
	uses cuda to filter out unprime numbers
	result : an array of prime numbers using CUDA powers :)
	
## Stage 2
	
   building RSA key using RSA algorithm, number e is chosen at random as a prime number below phi(n)
   
## Stage 3

   testing the key ;)
   
   encrypting and decrypting is using multi threading on openmp

Limited to long long size -> 8 bytes (64 bits) meaning 2^63 (signed unsigned bit required)   
real RSA is 4096 bits (512 bytes)

# Example of a run on small input:

CUDA satistics
 Amount of blocks = 1024, Threads per block =1024


Done with creating prime numbers list
 the numbers I was searching was between 0 and 200
 and I found 47 prime numbers


RSA key data:
p = 41, q = 53
n = 2173, phi(n) = 2080
d = 157, e = 53


Encryption data:

 Message: 145

 Encrypted Message: 887


Decryption data:

 Decrypted Message: 145, Message was 145


Entire operation of searching 47 prime numbers between 0 and 200
And generating RSA key
And encrypting
And decrypting
Took 1.000000 seconds
Press any key to continue . . .


# Example of a run on medium input:

CUDA satistics
 Amount of blocks = 1024, Threads per block =1024


Done with creating prime numbers list
 the numbers I was searching was between 0 and 65535
 and I found 6543 prime numbers


RSA key data:
p = 19477, q = 19501
n = 379820977, phi(n) = 379782000
d = 15190501, e = 19501


Encryption data:

 Message: 28385

 Encrypted Message: 219707150


Decryption data:

 Decrypted Message: 28385, Message was 28385


Entire operation of searching 6543 prime numbers between 0 and 65535
And generating RSA key
And encrypting
And decrypting
Took 19.000000 seconds
Press any key to continue . . .

# Example of a run on big input:

CUDA satistics
 Amount of blocks = 1024, Threads per block =1024


Done with creating prime numbers list
 the numbers I was searching was between 0 and 100000
 and I found 9593 prime numbers


RSA key data:
p = 60527, q = 60601
n = 3667996727, phi(n) = 3667875600
d = 48904201, e = 60601


Encryption data:

 Message: 6106

 Encrypted Message: 2453195187


Decryption data:

 Decrypted Message: 6106, Message was 6106


Entire operation of searching 9593 prime numbers between 0 and 100000
And generating RSA key
And encrypting
And decrypting
Took 44.000000 seconds
Press any key to continue . . .



