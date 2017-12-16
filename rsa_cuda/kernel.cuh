#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// 2^16.... n = 2^(16^2) at max, so 2^32
long long RANDOM_NUM_MAX = 65535;

typedef struct RSA
{
	long long n;
	long long e;
	long long d;
}RSA;


long long encrypt(RSA* key, long long message);

long long decrypt(RSA* key, long long crypted_message);

long long int m_pow_p_mod_n(long long m, long long p, long long n);


long long random_num(long long max); // random number between 0 and max


long long get_prime(); // random prime number

long long get_prime(long long max); // random prime number -  has to be smaller than max

RSA* get_rsa(); // gets an rsa key (public + private)

long long* list_prime = NULL; // list of prime numbers
long long list_prime_size = 0; // the list size

void init_list_prime(); // makes the list on the gpu, here we send commands from cpu

/*
	to staisfy d*e == 1 mod phi_n
*/
long long gcdExtended(long long a, long long b, long long *x, long long *y);

/*
	CUDA code to check if number is prime
*/
__global__ void init_is_prime_list(long long* num, long long i, long long j, int threads_count, int blocks_count);

/*
	GPU satistics
*/
int threads_max;
int blocks_per_thread;
void init_GPU_limits();

