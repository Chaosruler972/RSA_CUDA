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
   
   encrypting and decrypting is using multi threading on openmp, though, it's still bit-limited, a big integer will result in 
   miscalculated results (this is not python!)
   
# Example of a run:

p = 101, q = 113
n = 11413, phi(n) = 11200
d = 2577, e = 113

 Message: 386

 Encrypted Message: 8861

 Decrypted Message: 386
Press any key to continue . . .



