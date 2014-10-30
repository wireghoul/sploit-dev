from demangler import *

#starting/ending addresses for tag associations in mshtml (version: 9.0.8112.16464)
start_addr = 0x6403C528
end_addr = 0x6403E2B0

dm = Demangler()

curr_addr = start_addr
while (curr_addr < end_addr):

    #get the associate struture associated with each tag
    const_g = Dword(curr_addr) - 0x10
    assoc_name = Name(const_g)
    if assoc_name:
        const_s = dm.VCDemangle(assoc_name, True)                        
        print "AssocName: [%X] -> %s" % (const_g, const_s)
    curr_addr += 4

    #read the actual tag name (this is what will be in the html)
    tag_offset = Dword(curr_addr)
    tag_type = GetStringType(tag_offset)
    if tag_type:
        print "TagName: %s" % (GetString(tag_offset, -1, tag_type))
    curr_addr += 4

##    unk_struct1 = Dword(curr_addr)
##    unk_struct1_name = Name(unk_struct1)
##    print "Unkonwn Struct: %X %s" % (curr_addr, unk_struct1_name)
    curr_addr += 4

##    unk_struct2 = Dword(curr_addr)
##    unk_struct2_name = Name(unk_struct1)
##    print "Unkonwn Struct: %X %s" % (curr_addr, unk_struct2_name)
    curr_addr += 4

##    unk_dword1 = Dword(curr_addr)
##    print "DWORD1: %X %X" % (curr_addr, unk_dword1)
    curr_addr += 4

##    unk_dword2 = Dword(curr_addr)
##    print "DWORD2: %X %X" % (curr_addr, unk_dword2)
    curr_addr += 4   

    #a VERY crude demangler
    const_loc = Dword(curr_addr)
    const_mangled = GetFunctionName(const_loc)
    const_name = dm.VCDemangle(const_mangled, True)
    print "Constructor Name: %s @ 0x%X" % (const_name, const_loc)
    curr_addr += 4

##    const_loc2 = Dword(curr_addr)
##    const_mangled2 = GetFunctionName(const_loc2)
##    print "Mangled: %X %s" % (const_loc2, const_mangled2)
##    const_name2 = dm.VCDemangle(const_mangled2, True)
##    print "Constructor Name: %s" % (const_name2)
    curr_addr += 4

##    unk_dword3 = Dword(curr_addr)
##    print "DWORD3: %X %X" % (curr_addr, unk_dword3)
    curr_addr += 4     

    #go into constructor
    func_start = const_loc
    func_end = FindFuncEnd(const_loc)

    #a pretty poor method to walk backwards after finding a HeapAlloc call
    push_count = 0
    alloc_args = {0:'', 1:'', 2:''}
    for head in Heads(func_start, func_end):
        if isCode(GetFlags(head)):
            mnem = GetMnem(head)
            if mnem.lower() == "call":
                operand = GetOpnd(head, 0)
                if operand != BADADDR:
                    temp_head = head
                    seen_pop = False
                    if operand.find("ProcessHeapAllocClear") != -1:
                        while(True):
                            temp_head = PrevHead(temp_head, func_start)
                            mnem = GetMnem(temp_head)
                            if mnem.lower() == "mov":
                                reg = GetOpnd(temp_head, 0)
                                if reg != BADADDR and reg.lower() == "eax":
                                    cnt = GetOpnd(temp_head, 1)
                                    if cnt != BADADDR:
                                            alloc_args[0] = "_g_hProcessHeap"
                                            alloc_args[1] = 8
                                            alloc_args[2] = cnt.rstrip("h")
                                            break
                            elif mnem.lower() == "pop":
                                seen_pop = True
                            elif seen_pop:
                                if mnem.lower() == "push":
                                    cnt = GetOpnd(temp_head, 0)
                                    if cnt != BADADDR:
                                        alloc_args[0] = "_g_hProcessHeap"
                                        alloc_args[1] = 8
                                        alloc_args[2] = cnt.rstrip("h")
                                        break
                                        
                                
                    elif operand.find("HeapAlloc") != -1:
                        while(True):
                            if push_count > 2:
                                break;
                            
                            temp_head = PrevHead(temp_head, func_start)
                            mnem = GetMnem(temp_head)
                            if mnem.lower() == "push":
                                cnt = GetOpnd(temp_head, 0)
                                #print "Push[%X] => %s" % (push_count, cnt)
                                alloc_args[push_count] = cnt.rstrip("h")
                                push_count += 1
                        break

    heap = "None"
    flags = 0x0
    size = 0x0
    if(alloc_args[0] != '' and alloc_args[1] != '' and alloc_args[2] != ''):
        #print alloc_args
        heap =  alloc_args[0]
        flags = alloc_args[1]
        if not isinstance(flags, int):
            flags = int(flags, 16)
        size = alloc_args[2]
        if not isinstance(size, int):
            size = int(size, 16)

        print "HeapAlloc(%s, 0x%X, 0x%X)" % (heap, flags, size)
    else:
        print "No Heap Allocation"

    print "--------------------------------"
