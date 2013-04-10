# INSTRUCTIONS: save as ~/.gdbinit
#
# DESCRIPTION: A cracker-friendly gdb configuration file.
#
# REVISION : 6.1 
# 
# CONTRIBUTORS: mammon_, elaine, pusillus, mong
#
# FEEDBACK: http://board.anticrack.de/viewforum.php?f=35
# 
# NOTES: 'help user' in gdb will list the commands/descriptions in this file
#        'context on' now enables auto-display of context screen
#
# CHANGELOG:
# Version 6.1
#        fixed filename in step_to_call so it points to /dev/null
#        changed location of logfiles from /tmp  to ~
# Version 6
#  added print_insn_type, get_insn_type, context-on, context-off commands
#  added trace_calls, trace_run, step_to_call commands
#  changed hook-stop so it checks $SHOW_CONTEXT variable
# Version 5
#  added bpm, dump_bin, dump_hex, bp_alloc commands
#        added 'assemble' by elaine, 'gas_asm' by mong
#  added Tip Topics for aspiring crackers ;)
#   Version 4
#  added eflags-changing insns by pusillus
#  added bp, nop, null, and int3 patch commands, also hook-stop
# Version 3
#  incorporated elaine's if/else goodness into the hex/ascii dump
# Version 2
#  radix bugfix by elaine
# TODO:
# * add global vars to allow user to control stack,data,code win sizes
# * add dump, append, set write, etc commands
# * more tips!


# ______________breakpoint aliases_____________
define bpl
 info breakpoints
end
document bpl
List breakpoints
end

define bp
 set $SHOW_CONTEXT = 1
 break * $arg0
end
document bp
Set a breakpoint on address
Usage: bp addr
end

define bpc 
 clear $arg0
end
document bpc
Clear breakpoint at function/address
Usage: bpc addr
end

define bpe
 enable $arg0
end
document bpe
Enable breakpoint #
Usage: bpe num
end

define bpd
 disable $arg0
end
document bpd
Disable breakpoint #
Usage: bpd num
end

define bpt
 set $SHOW_CONTEXT = 1
 tbreak $arg0
end
document bpt
Set a temporary breakpoint on address
Usage: bpt addr
end

define bpm
 set $SHOW_CONTEXT = 1
 awatch $arg0
end
document bpm
Set a read/write breakpoint on address
Usage: bpm addr
end

define bhb
 set $SHOW_CONTEXT = 1
 hb * $arg0
end
document bhb
Set Hardware Assisted breakpoint on address
Usage: bhb addr
end

# ______________process information____________
define argv
 show args
end
document argv
Print program arguments
end

define stack
 info stack
end
document stack
Print call stack
end

define frame
 info frame
 info args
 info locals
end
document frame
Print stack frame
end

define flags
 if (($eflags >> 0xB) & 1 )
  printf "O "
 else
  printf "o "
 end
 if (($eflags >> 0xA) & 1 )
  printf "D "
 else
  printf "d "
 end
 if (($eflags >> 9) & 1 )
  printf "I "
 else
  printf "i "
 end
 if (($eflags >> 8) & 1 )
  printf "T "
 else
  printf "t "
 end
 if (($eflags >> 7) & 1 )
  printf "S "
 else
  printf "s "
 end
 if (($eflags >> 6) & 1 )
  printf "Z "
 else
  printf "z "
 end
 if (($eflags >> 4) & 1 )
  printf "A "
 else
  printf "a "
 end
 if (($eflags >> 2) & 1 )
  printf "P "
 else
  printf "p "
 end
 if ($eflags & 1)
  printf "C "
 else
  printf "c "
 end
 printf "\n"
end
document flags
Print flags register
end

define eflags
 printf "     OF <%d>  DF <%d>  IF <%d>  TF <%d>",\
        (($eflags >> 0xB) & 1 ), (($eflags >> 0xA) & 1 ), \
        (($eflags >> 9) & 1 ), (($eflags >> 8) & 1 )
 printf "  SF <%d>  ZF <%d>  AF <%d>  PF <%d>  CF <%d>\n",\
        (($eflags >> 7) & 1 ), (($eflags >> 6) & 1 ),\
        (($eflags >> 4) & 1 ), (($eflags >> 2) & 1 ), ($eflags & 1)
 printf "     ID <%d>  VIP <%d> VIF <%d> AC <%d>",\
        (($eflags >> 0x15) & 1 ), (($eflags >> 0x14) & 1 ), \
        (($eflags >> 0x13) & 1 ), (($eflags >> 0x12) & 1 )
 printf "  VM <%d>  RF <%d>  NT <%d>  IOPL <%d>\n",\
        (($eflags >> 0x11) & 1 ), (($eflags >> 0x10) & 1 ),\
        (($eflags >> 0xE) & 1 ), (($eflags >> 0xC) & 3 )
end
document eflags
Print entire eflags register
end

define reg
 printf "    eax:%08X  ebx:%08X  ecx:%08X ",  $eax, $ebx, $ecx
 printf " edx:%08X     eflags:%08X\n",  $edx, $eflags
 printf "    esi:%08X  edi:%08X  esp:%08X ",  $esi, $edi, $esp
 printf " ebp:%08X     eip:%08X\n", $ebp, $eip
 printf "     cs:%04X  ds:%04X  es:%04X", $cs, $ds, $es
 printf "  fs:%04X  gs:%04X  ss:%04X    ", $fs, $gs, $ss
 echo \033[31m
 flags
 echo \033[0m
end
document reg
Print CPU registers
end

define func
 info functions
end
document func
Print functions in target
end

define var
 info variables
end
document var
Print variables (symbols) in target
end

define lib
 info sharedlibrary
end
document lib
Print shared libraries linked to target
end

define sig
 info signals
end
document sig
Print signal actions for target
end

define thread
 info threads
end
document thread
Print threads in target
end

define u
 info udot
end
document u
Print kernel 'user' struct for target
end

define dis
 disassemble $arg0
end
document dis
Disassemble address
Usage: dis addr
end

# ________________hex/ascii dump an address______________
define ascii_char
 # thanks elaine :)
 set $_c=*(unsigned char *)($arg0)
 if ( $_c < 0x20 || $_c > 0x7E )
  printf "."
 else
  printf "%c", $_c
 end
end
document ascii_char
Print the ASCII value of arg0 or '.' if value is unprintable
end

define hex_quad
 printf "%02X %02X %02X %02X  %02X %02X %02X %02X",                          \
    *(unsigned char*)($arg0), *(unsigned char*)($arg0 + 1),      \
    *(unsigned char*)($arg0 + 2), *(unsigned char*)($arg0 + 3),  \
    *(unsigned char*)($arg0 + 4), *(unsigned char*)($arg0 + 5),  \
    *(unsigned char*)($arg0 + 6), *(unsigned char*)($arg0 + 7)
end
document hex_quad
Print eight hexadecimal bytes starting at arg0
end

define hexdump
 printf "%08X : ", $arg0
 hex_quad $arg0
 printf " - "
 hex_quad ($arg0+8)
 printf " "

 ascii_char ($arg0)
 ascii_char ($arg0+1)
 ascii_char ($arg0+2)
 ascii_char ($arg0+3)
 ascii_char ($arg0+4)
 ascii_char ($arg0+5)
 ascii_char ($arg0+6)
 ascii_char ($arg0+7)
 ascii_char ($arg0+8)
 ascii_char ($arg0+9)
 ascii_char ($arg0+0xA)
 ascii_char ($arg0+0xB)
 ascii_char ($arg0+0xC)
 ascii_char ($arg0+0xD)
 ascii_char ($arg0+0xE)
 ascii_char ($arg0+0xF)

 printf "\n"
end
document hexdump
Display a 16-byte hex/ASCII dump of arg0
end

# ________________data window__________________
define ddump
 echo \033[36m

 printf "[%04X:%08X]------------------------", $ds, $data_addr
 printf "---------------------------------[ data]\n"
 echo \033[0m
 set $_count=0
 while ( $_count < $arg0 )
  set $_i=($_count*0x10)
  hexdump ($data_addr+$_i)
  set $_count++
 end
end
document ddump
Display $arg0 lines of hexdump for address $data_addr
end

define dd
 if ( ($arg0 & 0x40000000) || ($arg0 & 0x08000000) || ($arg0 & 0xBF000000) )
  set $data_addr=$arg0
  ddump 0x10
 else
  printf "Invalid address: %08X\n", $arg0
 end
end
document dd
Display 16 lines of a hex dump for $arg0
end

define stackwin
 hexdump $ebp-0x0C
 hexdump $ebp-0x1C
 hexdump $ebp-0x2C
 hexdump $ebp-0x3C
end

define datawin
# if ( ($esi & 0x40000000) || ($esi & 0x08000000) || ($esi & 0xBF000000) )
#  set $data_addr=$esi
# else
#  if ( ($edi & 0x40000000) || ($edi & 0x08000000) || ($edi & 0xBF000000) )
#   set $data_addr=$edi
#  else
#   if ( ($eax & 0x40000000) || ($eax & 0x08000000) || \
#        ($eax & 0xBF000000) )
#
#    set $data_addr=$eax
#   else
#    set $data_addr=$esp
#   end
#  end
# end
 ddump 8
end
document datawin
Display esi, edi, eax, or esp in data window
end

# ________________process context______________
define context
 echo \033[36m
 printf "----------------------------------------"
 printf "---------------------------------[ regs]\n"
 echo \033[32m
 reg
 echo \033[36m
 printf "[%04X:%08X]------------------------", $ss, $esp
 printf "---------------------------------[stack]\n"
 echo \033[0m
 stackwin
 datawin
 echo \033[36m
 printf "[%04X:%08X]------------------------", $cs, $eip
 printf "---------------------------------[ code]\n"
 echo \033[37m
 x /8i $pc
 echo \033[36m
 printf "---------------------------------------"
 printf "----------------------------------------\n"
 echo \033[0m
end
document context
Print regs, stack, ds:esi, and disassemble cs:eip
end

define context-on
 set $SHOW_CONTEXT = 1
end
document context-on
Enable display of context on every program stop
end

define context-off
 set $SHOW_CONTEXT = 1
end
document context-on
Disable display of context on every program stop
end

# ________________process control______________
define n
 ni
end
document n
Step one instruction
end

define go
 stepi $arg0
end
document go
Step # instructions
end

define pret
 finish
end
document pret
Step out of current call
end

define init
 set $SHOW_CONTEXT = 1
 set $SHOW_NEST_INSN=0
 tbreak _init
 r
end
document init
Run program; break on _init()
end

define start
 set $SHOW_CONTEXT = 1
 set $SHOW_NEST_INSN=0
 tbreak _start
 r
end
document start
Run program; break on _start()
end

define sstart
 set $SHOW_CONTEXT = 1
 set $SHOW_NEST_INSN=0
 tbreak __libc_start_main
 r
end
document sstart
Run program; break on __libc_start_main(). Useful for stripped executables.
end

define main
 set $SHOW_CONTEXT = 1
 set $SHOW_NEST_INSN=0
 tbreak main
 r
 set $data_addr=$pc
end
document main
Run program; break on main()
end

# ________________eflags commands_______________
define cfc
 if ($eflags & 1)
  set $eflags = $eflags&~1
 else
  set $eflags = $eflags|1
 end
end
document cfc
change Carry Flag
end

define cfp
 if (($eflags >> 2) & 1 )
  set $eflags = $eflags&~0x4
 else
  set $eflags = $eflags|0x4
 end
end
document cfp
change Carry Flag
end

define cfa
 if (($eflags >> 4) & 1 )
  set $eflags = $eflags&~0x10
 else
  set $eflags = $eflags|0x10
 end
end
document cfa
change Auxiliary Carry Flag
end

define cfz
 if (($eflags >> 6) & 1 )
  set $eflags = $eflags&~0x40
 else
  set $eflags = $eflags|0x40
 end
end
document cfz
change Zero Flag
end

define cfs
 if (($eflags >> 7) & 1 )
  set $eflags = $eflags&~0x80
 else
  set $eflags = $eflags|0x80
 end
end
document cfs
change Sign Flag
end

define cft
 if (($eflags >>8) & 1 )
  set $eflags = $eflags&100
 else
  set $eflags = $eflags|100
 end
end
document cft
change Trap Flag
end

define cfi
 if (($eflags >> 9) & 1 )
  set $eflags = $eflags&~0x200
 else
  set $eflags = $eflags|0x200
 end
end
document cfi
change Interrupt Flag
end

define cfd
 if (($eflags >>0xA ) & 1 )
  set $eflags = $eflags&~0x400
 else
  set $eflags = $eflags|0x400
 end
end
document cfd
change Direction Flag
end

define cfo
 if (($eflags >> 0xB) & 1 )
  set $eflags = $eflags&~0x800
 else
  set $eflags = $eflags|0x800
 end
end
document cfo
change Overflow Flag
end

# --------------------patch---------------------
define nop
 set * (unsigned char *) $arg0 = 0x90
end
document nop
Patch byte at address arg0 to a NOP insn
Usage: nop addr
end

define null
 set * (unsigned char *) $arg0 = 0
end
document null
Patch byte at address arg0 to  NULL
Usage: null addr
end

define int3
 set * (unsigned char *) $arg0 = 0xCC
end
document int3
Patch byte at address arg0 to an INT3 insn
Usage: int3 addr
end

# --------------------cflow---------------------
define print_insn_type
 if ($arg0 == 0)
  printf "UNKNOWN";
 end
 if ($arg0 == 1)
  printf "JMP";
 end
 if ($arg0 == 2)
  printf "JCC";
 end
 if ($arg0 == 3)
  printf "CALL";
 end
 if ($arg0 == 4)
  printf "RET";
 end
 if ($arg0 == 5)
  printf "INT";
 end
end
document print_insn_type
This prints the human-readable mnemonic for the instruction typed passed as
a parameter (usually $INSN_TYPE).
end

define get_insn_type
 set $INSN_TYPE = 0
 set $_byte1=*(unsigned char *)$arg0
 if ($_byte1 == 0x9A || $_byte1 == 0xE8 )
  # "call"
  set $INSN_TYPE=3
 end
 if ($_byte1 >= 0xE9 && $_byte1 <= 0xEB)
  # "jmp"
  set $INSN_TYPE=1
 end
 if ($_byte1 >= 0x70 && $_byte1 <= 0x7F)
  # "jcc"
  set $INSN_TYPE=2
 end
 if ($_byte1 >= 0xE0 && $_byte1 <= 0xE3 )
  # "jcc"
  set $INSN_TYPE=2
 end
 if ($_byte1 == 0xC2 || $_byte1 == 0xC3 || $_byte1 == 0xCA || $_byte1 == 0xCB || $_byte1 == 0xCF)
  # "ret"
  set $INSN_TYPE=4    
 end
 if ($_byte1 >= 0xCC && $_byte1 <= 0xCE)
  # "int"
  set $INSN_TYPE=5
 end
 if ($_byte1 == 0x0F )
  # two-byte opcode
  set $_byte2=*(unsigned char *)($arg0 +1)
  if ($_byte2 >= 0x80 && $_byte2 <= 0x8F)
   # "jcc"
   set $INSN_TYPE=2
  end
 end
 if ($_byte1 == 0xFF )  
  # opcode extension
  set $_byte2=*(unsigned char *)($arg0 +1)
  set $_opext=($_byte2 & 0x38)
  if ($_opext == 0x10 || $_opext == 0x18) 
   # "call"
   set $INSN_TYPE=3
  end
  if ($_opext == 0x20 || $_opext == 0x28)
   # "jmp"
   set $INSN_TYPE=1
  end
 end
end
document get_insn_type
This takes an address as a parameter and sets the global $INSN_TYPE variable
to 0, 1, 2, 3, 4, 5 if the instruction at the address is unknown, a jump,
a conditional jump, a call, a return, or an interrupt.
end

define step_to_call
 set $_saved_ctx = $SHOW_CONTEXT
 set $SHOW_CONTEXT = 0
 set $SHOW_NEST_INSN=0
 set logging file /dev/null
 set logging on
 set logging redirect on
 set $_cont = 1

 while ( $_cont > 0 )
  stepi
  get_insn_type $pc
  if ($INSN_TYPE == 3)
   set $_cont = 0
  end
 end

 if ( $_saved_ctx > 0 )
  context
 else
  x /i $pc
 end

 set $SHOW_CONTEXT = 1
 set $SHOW_NEST_INSN=0
 set logging redirect off
 set logging off
 set logging file gdb.txt
end
document step_to_call
This single steps until it encounters a call instruction; it stops before
the call is taken.
end

define trace_calls
 set $SHOW_CONTEXT = 0
 set $SHOW_NEST_INSN=0
 set $_nest = 1
 set listsize 0
 set logging overwrite on
 set logging file ~/gdb_trace_calls.txt
 set logging on
 set logging redirect on
 
 while ( $_nest > 0 )
  get_insn_type $pc

  # handle nesting
  if ($INSN_TYPE == 3)
   set $_nest = $_nest + 1
  else
   if ($INSN_TYPE == 4)
    set $_nest = $_nest - 1
   end
  end

  # if a call, print it
  if ($INSN_TYPE == 3)
   set $x = $_nest
   while ( $x > 0 )
    printf "\t"
    set $x = $x - 1
   end
   x /i $pc
  end

  #set logging file /dev/null
  stepi
  #set logging file ~/gdb_trace_calls.txt
 end

 set $SHOW_CONTEXT = 1
 set $SHOW_NEST_INSN=0
 set logging redirect off
 set logging off
 set logging file gdb.txt
 
 # clean up trace file
 shell  grep -v ' at ' ~/gdb_trace_calls.txt > ~/gdb_trace_calls.1
 shell  grep -v ' in ' ~/gdb_trace_calls.1 > ~/gdb_trace_calls.txt
end
document trace_calls
Creates a runtime trace of the calls made target in ~/gdb_trace_calls.txt. 
Note that this is very slow because gdb "set redirect on" does not work!
end

define trace_run
 set $SHOW_CONTEXT = 0
 set $SHOW_NEST_INSN=1
 set logging overwrite on
 set logging file ~/gdb_trace_run.txt
 set logging on
 set logging redirect on
 set $_nest = 1

 while ( $_nest > 0 )

  get_insn_type $pc
  # jmp, jcc, or cll
  if ($INSN_TYPE == 3)
   set $_nest = $_nest + 1
  else
   # ret
   if ($INSN_TYPE == 4)
    set $_nest = $_nest - 1
   end
  end

  stepi
 end

 set $SHOW_CONTEXT = 1
 set $SHOW_NEST_INSN=0
 set logging file gdb.txt
 set logging redirect off
 set logging off

 # clean up trace file
 shell  grep -v ' at ' ~/gdb_trace_run.txt > ~/gdb_trace_run.1
 shell  grep -v ' in ' ~/gdb_trace_run.1 > ~/gdb_trace_run.txt

end
document trace_run
Creates a runtime trace of the target in ~/gdb_trace_run.txt. Note
that this is very slow because gdb "set redirect on" does not work!
end


# _____________________misc_____________________
# this makes 'context' be called at every BP/step
define hook-stop
 if ( $SHOW_CONTEXT > 0 )
  context
 end
 if ( $SHOW_NEST_INSN > 0 )
  set $x = $_nest
  while ($x > 0 )
   printf "\t"
   set $x = $x - 1
  end
 end
end

define assemble
printf "Hit Ctrl-D to start, type code to assemble, hit Ctrl-D when done.\n"
printf "It is recommended to start with\n"
printf "\tBITS 32\n"
printf "Note that this command uses NASM (Intel syntax) to assemble.\n"
shell nasm -f bin -o /dev/stdout /dev/stdin | od -v -t x1 -w16 -A n
end
document assemble
Assemble Intel x86 instructions to binary opcodes. Uses nasm.
Usage: assemble
end

define gas_asm
printf "Type code to assemble, hit Ctrl-D until results appear :)\n"
printf "Note that this command uses GAS (AT&T syntax) to assemble.\n"
shell as -o ~/__gdb_tmp.bin
shell objdump -d -j .text --adjust-vma=$arg0 ~/__gdb_tmp.bin 
shell rm ~/__gdb_tmp.bin
end
document gas_asm
Assemble Intel x86 instructions to binary opcodes using gas and objdump
Usage: gas_asm address
end

# !scary bp_alloc macro!
# The idea behind this macro is to break on the following code:
# 0x4008e0aa <malloc+6>:  sub    $0xc,%esp
# 0x4008e0ad <malloc+9>:  call   0x4008e0b2 <malloc+14>
# 0x4008e0b2 <malloc+14>: pop    %ebx
# 0x4008e0b3 <malloc+15>: add    $0xa3f6e,%ebx
# At 0x4008e0b3, %ebx contains the address that has just been allocated
# The bp_alloc macro generates this breakpoint and *should* work for
# the forseeable future ... but if it breaks, set a breakpoint on 
# __libc_malloc and look for where where the return value gets popped.

define bp_alloc
 tbreak *(*__libc_malloc + F) if $ebx == $arg0
end
document bp_alloc
This sets a temporary breakpoint on the allocation of $arg0.
It works by setting a breakpoint on a specific address in __libc_malloc().
USE WITH CAUTION -- it is extremely platform dependent.
Usage: bp_alloc addr
end

define dump_hexfile
 dump ihex memory $arg0 $arg1 $arg2
end
document dump_hexfile
Write a range of memory to a file in Intel ihex (hexdump) format.
Usage:  dump_hexfile filename start_addr end_addr
end

define dump_binfile
 dump memory $arg0 $arg1 $arg2
end
document dump_binfile
Write a range of memory to a binary file.
Usage:  dump_binfile filename start_addr end_addr
end
  
# _________________cracker tips_________________
# The 'tips' command is used to provide tutorial-like info to the user
define tips
  printf "Tip Topic Commands:\n"
  printf "\ttip_display : Automatically display values on each break\n"
  printf "\ttip_patch   : Patching binaries\n"
  printf "\ttip_strip   : Dealing with stripped binaries\n"
  printf "\ttip_syntax  : ATT vs Intel syntax\n"
  printf "\ttip_custom  : my tips :D\n"
end
document tips
Provide a list of tips from crackers on various topics. 
end

define tip_custom
  printf "\n"
  printf "to see dump"
  printf "$set data_addr=XXX"
  printf "\n"
  printf "to display ASM code"
  printf "display /6i $pc"
  printf "\n"
end

define tip_patch
  printf "\n"
  printf "                   PATCHING MEMORY\n"
  printf "Any address can be patched using the 'set' command:\n"
  printf "\t`set ADDR = VALUE` \te.g. `set *0x8049D6E = 0x90`\n"
  printf "\n"
  printf "                 PATCHING BINARY FILES\n"
  printf "Use `set write` in order to patch the target executable\n"
  printf "directly, instead of just patching memory.\n"
  printf "\t`set write on` \t`set write off`\n"
  printf "Note that this means any patches to the code or data segments\n"
  printf "will be written to the executable file. When either of these\n"
  printf "commands has been issued, the file must be reloaded.\n"
  printf "\n"
end
document tip_patch
Tips on patching memory and binary files
end

define tip_strip
  printf "\n"
  printf "             STOPPING BINARIES AT ENTRY POINT\n"
  printf "Stripped binaries have no symbols, and are therefore tough to\n"
  printf "start automatically. To debug a stripped binary, use\n"
  printf "\tinfo file\n"
  printf "to get the entry point of the file. The first few lines of\n"
  printf "output will look like this:\n"
  printf "\tSymbols from '/tmp/a.out'\n"
  printf "\tLocal exec file:\n"
  printf "\t        `/tmp/a.out', file type elf32-i386.\n"
  printf "\t        Entry point: 0x80482e0\n"
  printf "Use this entry point to set an entry point:\n"
  printf "\t`tbreak *0x80482e0`\n"
  printf "The breakpoint will delete itself after the program stops as\n"
  printf "the entry point.\n"
  printf "\n"
end
document tip_strip
Tips on dealing with stripped binaries
end

define tip_syntax
  printf "\n"
  printf "\t    INTEL SYNTAX                        AT&T SYNTAX\n"
  printf "\tmnemonic dest, src, imm            mnemonic src, dest, imm\n" 
  printf "\t[base+index*scale+disp]            disp(base, index, scale)\n"
  printf "\tregister:      eax                 register:      %%eax\n"
  printf "\timmediate:     0xFF                immediate:     $0xFF\n"
  printf "\tdereference:   [addr]              dereference:   addr(,1)\n"
  printf "\tabsolute addr: addr                absolute addr: *addr\n"
  printf "\tbyte insn:     mov byte ptr        byte insn:     movb\n"
  printf "\tword insn:     mov word ptr        word insn:     movw\n"
  printf "\tdword insn:    mov dword ptr       dword insn:    movd\n"
  printf "\tfar call:      call far            far call:      lcall\n"
  printf "\tfar jump:      jmp far             far jump:      ljmp\n"
  printf "\n"
  printf "Note that order of operands in reversed, and that AT&T syntax\n"
  printf "requires that all instructions referencing memory operands \n"
  printf "use an operand size suffix (b, w, d, q).\n"
  printf "\n"
end
document tip_syntax
Summary of Intel and AT&T syntax differences
end

define tip_display
printf "\n"
printf "Any expression can be set to automatically be displayed every time\n"
printf "the target stops. The commands for this are:\n"
printf "\t`display expr'     : automatically display expression 'expr'\n"
printf "\t`display'          : show all displayed expressions\n"
printf "\t`undisplay num'    : turn off autodisplay for expression # 'num'\n"
printf "Examples:\n"
printf "\t`display/x *(int *)$esp`      : print top of stack\n"
printf "\t`display/x *(int *)($ebp+8)`  : print first parameter\n"
printf "\t`display (char *)$esi`        : print source string\n"
printf "\t`display (char *)$edi`        : print destination string\n"
printf "\n"
end
document tip_display
Tips on automatically displaying values when a program stops.
end
# __________________gdb options_________________
set confirm off
set verbose off
#set prompt \033[01;m\033] niel@gdb $ \033[0m
set prompt \033[31mgdb $ \033[0m
set output-radix 0x10
set input-radix 0x10
# These make gdb never pause in its output
set height 0
set width 0
# why do these not work???
set $SHOW_CONTEXT = 1
set $SHOW_NEST_INSN=0

set disassembly-flavor intel


#EOF

