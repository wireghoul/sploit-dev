// SHA1 kernel sandbox

/**
 
 BFI_INT is equivalent to the (bit)vector_select function in this optimization:

(0 ≤ i ≤ 19): f = vec_sel(d, c, b)

But I also found another optimization where it is useful

(40 ≤ i ≤ 59): f = vec_sel(b, (c or d), (c and d))

This brings this round function down to 3 cycles, which is faster than any of the alternatives listed on the Wikipedia page.
 
 * http://devgurus.amd.com/thread/143488
 */


int main() {
    
}