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

# Example of a run:

p = 499, q = 523
n = 260977, phi(n) = 259956
d = 124759, e = 523

 Message: 2159

 Encrypted Message: 189393

 Decrypted Message: 2159
Press any key to continue . . .



