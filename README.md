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
 the numbers I was searching was between 0 and 1000
 and I found 118 prime numbers


RSA key data:
p = 151, q = 173
n = 26123, phi(n) = 25800
d = 2237, e = 173


Encryption data:

 Message: 668

 Encrypted Message: 21601


Decryption data:

 Decrypted Message: 668, Message was 668


Entire operation of searching 118 prime numbers between 0 and 1000
And generating RSA key
And encrypting
And decrypting
Took 1.000000 seconds
Press any key to continue . . .




CUDA satistics
 Amount of blocks = 1024, Threads per block =1024


Done with creating prime numbers list
 the numbers I was searching was between 0 and 10000
 and I found 216 prime numbers


RSA key data:
p = 1973, q = 1997
n = 3940081, phi(n) = 3936112
d = 1259477, e = 1997


Encryption data:

 Message: 1802

 Encrypted Message: 249430


Decryption data:

 Decrypted Message: 1802, Message was 1802


Entire operation of searching 216 prime numbers between 0 and 10000
And generating RSA key
And encrypting
And decrypting
Took 4.000000 seconds
Press any key to continue . . .




CUDA satistics
 Amount of blocks = 1024, Threads per block =1024


Done with creating prime numbers list
 the numbers I was searching was between 0 and 65535
 and I found 1140 prime numbers


RSA key data:
p = 38261, q = 21407
n = 819053227, phi(n) = 818993560
d = 383959423, e = 21407


Encryption data:

 Message: 12004

 Encrypted Message: 595019569


Decryption data:

 Decrypted Message: 12004, Message was 12004


Entire operation of searching 1140 prime numbers between 0 and 65535
And generating RSA key
And encrypting
And decrypting
Took 25.000000 seconds
Press any key to continue . . .


