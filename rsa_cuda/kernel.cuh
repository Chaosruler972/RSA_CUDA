#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

typedef struct RSA
{
	unsigned int n;
	unsigned int e;
	unsigned int d;
}RSA;


long encrypt(RSA* key, long message);

long decrypt(RSA* key, long crypted_message);

long int m_pow_p_mod_n(long m, long p, unsigned int n);


unsigned int random_num(unsigned int max); // random number between 0 and max
// 2^16.... n = 2^(16^2) at max, so 2^32
#define RANDOM_NUM_MAX 256 

unsigned int get_prime(); // random prime number

unsigned int get_prime(unsigned int max); // random prime number -  has to be smaller than max

RSA* get_rsa(); // gets an rsa key (public + private)

unsigned int* list_prime = NULL; // list of prime numbers
unsigned int list_prime_size = 0; // the list size

void init_list_prime(); // makes the list on the gpu, here we send commands from cpu

/*
	to staisfy d*e == 1 mod phi_n
*/
unsigned int gcdExtended(unsigned int a, unsigned int b, unsigned int *x, unsigned int *y);

/*
	CUDA code to check if number is prime
*/
__global__ void init_is_prime_list(unsigned int* num, unsigned int i, unsigned int j, int threads_count, int blocks_count);

/*
	GPU satistics
*/
int threads_max;
int blocks_per_thread;
void init_GPU_limits();

