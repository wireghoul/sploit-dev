/**
 * This code is heavily based on the Python BFIPatcher.py script that is part
 * of the Phoenix Bitcoin miner.  Reference code used was available here on
 * March 5 2012:
 * http://svn3.xp-dev.com/svn/phoenix-miner/trunk/kernels/phatk/BFIPatcher.py
 * 
 * Python script license information, which I will continue to apply to this 
 * port into C++:
# Copyright (C) 2011 by jedi95 <jedi95@gmail.com> and 
#                       CFSworks <CFSworks@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
 * 
 * C++ port of code Copyright (C) 2012 Bitweasil <bitweasil@gmail.com> 
 */


// C defines
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// C++ defines
#include <vector>
#include <string>


#pragma pack(push)
#pragma pack(1)

// Structure for the main ELF header
typedef struct BFIELFHeader {
    uint64_t ident1;
    uint64_t ident2;
    
    uint16_t type;
    uint16_t machine;
    
    uint32_t version;
    uint32_t entry;
    uint32_t phoff;
    uint32_t shoff;
    uint32_t flags;
    
    uint16_t ehsize;
    uint16_t phentsize;
    uint16_t phnum;
    uint16_t shentsize;
    uint16_t shnum;
    uint16_t shstrndx;
} BFIELFHeader;

// Structure for the ELF subsections
typedef struct BFIELFSectionHeader {
    uint32_t nameIdx;
    uint32_t type;
    uint32_t flags;
    uint32_t addr;
    uint32_t offset;
    uint32_t size;
    uint32_t link;
    uint32_t info;
    uint32_t addralign;
    uint32_t entsize;
} BFIELFSectionHeader;
#pragma pack(pop)

// Structure for the needed section information
typedef struct SectionInfo {
    std::string name;
    uint64_t offset;
    uint64_t size;
} SectionInfo;


/**
 * The BFIPatcher class is a direct port of the BFIPatcher.py script that ships
 * with the Phoenix bitcoin miner.  This is the most comprehensive way of doing
 * the patching, as it properly decodes the ELF file and finds the correct
 * sections.  The other C/C++ methods (my old method, copied from gat3way's 
 * hashkill) are prone to false positives, and requires some empirical testing
 * to find the right offsets.  The new method parses the ELF-in-ELF, finds the
 * right .text segment, and performs the byte_align to BFI_INT translation
 * with the standard method.  Except it's all aligned and doesn't have the false
 * positive problem!  Yay!
 * 
 * Use: Get the binaries out of the OpenCL context with the
 * clGetProgramInfo(..., CL_PROGRAM_BINARIES, ...) command.
 * 
 * Convert the returned block of data into a vector, pass it in to the
 * patchBinary function, take the returned vector, and pass this data into the
 * clCreateProgramWithBinary() function.
 *
 * The GRTOpenCL class has an example of this usage.
 * 
 * This class uses the "fatal error means exit" approach.  If something is
 * unexpected, incorrect, wrong, etc, this will call exit(1) with an error
 * message.  If the binary is to use bfi_int patching and the bfi_int patching
 * cannot be performed, the results will be invalid/wrong anyway, so returning
 * the unpatched binary makes no sense here.
 * 
 */
class BFIPatcher {
public:
    /**
     * Perform the binary patching.  This function takes a vector of the 
     * ATI OpenCL ELF-in-ELF, patches all the byte_align instructions to be
     * bfi_int instructions in the standard method, and returns the patched
     * binary code.
     * 
     * @param newBinary The vector containing the OpenCL binary object
     * @return A modified copy of the vector with the byte_align patched to bfi_int
     */
    std::vector<uint8_t> patchBinary(std::vector<uint8_t> unpatchedBinary);
private:
    
    // The binary blob - unpatched until the end.
    std::vector<uint8_t> OpenCLBinary;
    
    // The ELF sections, extracted from the inner ELF
    std::vector<SectionInfo> ElfSections;
    
    // Inner ELF starting offset
    uint64_t innerELFOffset;

    /**
     * Reads through the provided binary and looks for the second ELF image.
     * @return The offset, in bytes, of the start of the second ELF tag.
     */
    uint64_t locateInner();
    
    /**
     * Reads the ELF subsections, populates the ElfSections vector.
     */
    void readELFSections();
    
    /**
     * Performs the actual byte_align to BFI_INT patching given the offset of
     * a section and the size of the section.
     * 
     * @param offset Start offset of the section within the inner ELF
     * @param size Size of the section
     */
    void patchOpcodes(uint64_t offset, uint64_t size);
    
};