
#include "kernel.cuh"


int main()
{
	init_GPU_limits();
	init_list_prime();
	//printf("threads_max = %d, blocks_per_thread=%d\n", threads_max, blocks_per_thread);
	
	RSA* key = get_rsa();
	
	long message = random_num(RANDOM_NUM_MAX);
	printf("\n Message: %ld\n", message);

	
	long encrypted_message = encrypt(key,message);
	printf("\n Encrypted Message: %ld\n", encrypted_message);
	
	long decrypted_message = decrypt(key, encrypted_message);
	printf("\n Decrypted Message: %ld\n", decrypted_message);
	
	free(key);
	free(list_prime);
	return 0;
}


/*
	o(n) on cpu level
*/
RSA* get_rsa()
{
	RSA* rsa = (RSA*)calloc(1,sizeof(RSA));
	unsigned int p = get_prime();
	unsigned int q = get_prime();
	while (p == q)
		q = get_prime(); // ensures p and q to be different prime numbers
	unsigned int n = p * q;
	rsa->n = n;
	unsigned int phi_n = (p - 1)*(q - 1);
	printf("p = %u, q = %u\nn = %u, phi(n) = %u\n", p, q, n, phi_n); 
	unsigned e = get_prime(phi_n);
	unsigned int x, y;
	gcdExtended(phi_n, e, &x, &y);
	unsigned int d = y;
	printf("d = %u, e = %u\n", d,e);
	rsa->d = d;
	rsa->e = e;
	return rsa;
}

__global__ void init_is_prime_list(unsigned int* num, unsigned int i, unsigned int j, int threads_count, int blocks_count)
{
	unsigned int num_to_check = (threadIdx.x)+threads_count*i;
	unsigned int divisor = (blockIdx.x) + blocks_count*j;
	if (num_to_check <= 1 || divisor <= 1 || num_to_check <= divisor )
		return;
	if (num_to_check%divisor == 0)
	{
		num[num_to_check] = divisor;
	}
	
}
/*
	o(logn) on GPU level -> o(n) on cpu level
*/
void init_list_prime()
{
	list_prime = (unsigned int*) calloc(RANDOM_NUM_MAX, sizeof(unsigned int));

	unsigned int *host_list_prime = NULL,*device_list_prime=NULL; // init data
	unsigned int i,j;

	cudaMallocHost((void**)&host_list_prime, sizeof(unsigned int)*RANDOM_NUM_MAX); // malloc 

	for (i = 0; i < RANDOM_NUM_MAX; i++)
		host_list_prime[i] = list_prime[i]; // copy to host pointer bridge

	cudaMalloc((void**)&device_list_prime, sizeof(unsigned int)*RANDOM_NUM_MAX); // alloc device pointer
	cudaMemcpy(device_list_prime, host_list_prime, sizeof(unsigned int)*RANDOM_NUM_MAX, cudaMemcpyHostToDevice); // copy to device pointer
	for (i = 0; i*threads_max <= RANDOM_NUM_MAX; i++)
	{
		for (j = 0; j*blocks_per_thread <= RANDOM_NUM_MAX; j++)
		{
			init_is_prime_list << < threads_max, blocks_per_thread >> > (device_list_prime,i,j,threads_max, blocks_per_thread);
			cudaDeviceSynchronize();
			cudaThreadSynchronize();
		}
	}
	

	cudaDeviceSynchronize();

	cudaMemcpy(host_list_prime, device_list_prime, sizeof(unsigned int)*RANDOM_NUM_MAX, cudaMemcpyDeviceToHost);
	// copy back only prime numbers
	unsigned int size = 0; // count amount of elements which are prime
	for (i = 1; i < RANDOM_NUM_MAX; i++)
	{
		if (host_list_prime[i] == 0) // if element is prime
			size++; // increase size
	}

	free(list_prime); // free current arr, it was big to bench the system for big memory/max memory usage
	list_prime = (unsigned int*)calloc(size, sizeof(unsigned int)); // allocates new array to hold list of prime numbers 
	
	cudaDeviceSynchronize();

	size = 0; // to iterate using similar manner
	for (i = 1; i < RANDOM_NUM_MAX; i++)
	{
		if (host_list_prime[i] == 0) // if number is prime
		{
			list_prime[size] = i; // set it on the prime array
			size++; // next iteration
		}
	}
	list_prime_size = size; // array size;
	cudaThreadSynchronize();
	cudaFreeHost(host_list_prime);
	cudaFree(device_list_prime);
}

/*
	o(gpu_count) .. o(n)...
*/
void init_GPU_limits()
{
	int deviceCount, device;
	//int gpuDeviceCount = 0;
	struct cudaDeviceProp properties;
	cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
	if (cudaResultCode != cudaSuccess)
		deviceCount = 0;
	/* machines with no GPUs can still report one emulation device */
	for (device = 0; device < deviceCount; ++device) {
		cudaGetDeviceProperties(&properties, device);
		if (properties.major != 9999) /* 9999 means emulation only */
			if (device == 0)
			{
				threads_max = properties.maxThreadsDim[0]/2;
				blocks_per_thread = properties.maxThreadsPerBlock/2;
			}
	}
}

/*
  o(1)
*/
unsigned int random_num(unsigned int max)
{
	srand((unsigned int)time(NULL));
	return (rand() % max) + 1;
}


/*
	o(1)
*/
unsigned int get_prime()
{
	unsigned int num = random_num(list_prime_size);
	return list_prime[num];
}

/*
	o(?) -- chooses random, depends how good seed is
*/
unsigned int get_prime(unsigned int max)
{
	unsigned int prime;
	while ((prime = get_prime()) >= max);
	return prime;
}

/*
	o( log(a*b) ) ..
*/
unsigned int gcdExtended(unsigned int a, unsigned int b, unsigned int *x, unsigned int *y)
{
	// Base Case
	if (a == 0)
	{
		*x = 0;
		*y = 1;
		return b;
	}

	unsigned int x1, y1; // To store results of recursive call
	int gcd = gcdExtended(b%a, a, &x1, &y1);

	// Update x and y using results of recursive
	// call
	*x = y1 - (b / a) * x1;
	*y = x1;

	return gcd;
}

long encrypt(RSA* key, long message)
{
	if (key == NULL)
		return 0;
	return  m_pow_p_mod_n(message,(long) key->e,key->n);
}

long decrypt(RSA* key, long crypted_message)
{
	if (key == NULL)
		return 0;
	return m_pow_p_mod_n(crypted_message,(long) key->d,key->n);
}

/*
	openMP multi threadded pow... hope this works
*/
long int m_pow_p_mod_n(long m, long p, unsigned int n)
{
	long res = 1;
	int trds = (omp_get_num_procs() > p ) ? p : omp_get_num_procs();
	#pragma omp parallel for num_threads(trds)
	for (register unsigned int i = 0; i < p; i++)
	{
		res *= m;
		res = res % n;
	}
	return res;
}