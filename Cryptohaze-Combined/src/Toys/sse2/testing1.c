#include <stdio.h>
#include <smmintrin.h>

int main ()
{
    __m128i a, b;

    a.m128i_u32[0] = 0;
    a.m128i_u32[1] = 65000;
    a.m128i_u32[2] = 128;
    a.m128i_u32[3] = 200000;

    b.m128i_u32[0] = 1;
    b.m128i_u32[1] = 70000;
    b.m128i_u32[2] = 127;
    b.m128i_u32[3] = 100000;

    __m128i res = _mm_max_epu32(a, b);

    printf_s("     a\t     b\t   res\n");
    printf_s("%6d\t%6d\t%6d\n%6d\t%6d\t%6d\n%6d\t%6d\t%6d\n%6d\t%6d\t%6d\n",
                a.m128i_u32[0], b.m128i_u32[0], res.m128i_u32[0],
                a.m128i_u32[1], b.m128i_u32[1], res.m128i_u32[1],
                a.m128i_u32[2], b.m128i_u32[2], res.m128i_u32[2],
                a.m128i_u32[3], b.m128i_u32[3], res.m128i_u32[3]);

    return 0;
}
