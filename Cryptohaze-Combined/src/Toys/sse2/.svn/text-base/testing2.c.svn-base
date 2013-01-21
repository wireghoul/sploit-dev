#include <xmmintrin.h>
 
extern void printv(__m128 m);
 
int main()
{
	__m128 m = _mm_set_ps(4, 3, 2, 1);
	__m128 z = _mm_setzero_ps();
 
	printv(m);
	printv(z);
 
	return 0;
}

