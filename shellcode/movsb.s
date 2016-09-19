BITS 32
xor ecx, ecx
mov ch, 0x10
xchg edi, esi
rep movsb
