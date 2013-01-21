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


// Change this line to point to your BFIPatcher.h file if needed!
#include "OpenCL_Common/BFIPatcher.h"

// Change to 1 for the local unit test
#define UNIT_TEST 0

// Set to 1 for debugging output.  Otherwise class is silent except for errors.
#define BFI_DEBUG 0

#if BFI_DEBUG
#define bfi_printf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#else
#define bfi_printf(fmt, ...) do {} while (0)
#endif



/**
 * Local utility function for this file to print the ELF header.
 * @param header Pointer to the ELF header to print
 */
static void printElfHeader(BFIELFHeader *header) {
    printf("ELF header\n");
    printf("ident1:    %016lx\n", header->ident1);
    printf("ident2:    %016lx\n", header->ident2);
    
    printf("type:      %04x\n", header->type);
    printf("machine:   %04x\n", header->machine);
    
    printf("version:   %08x\n", header->version);
    printf("entry:     %08x\n", header->entry);
    printf("phoff:     %08x\n", header->phoff);
    printf("shoff:     %08x\n", header->shoff);
    printf("flags:     %08x\n", header->flags);
    
    printf("ehsize:    %04x\n", header->ehsize);
    printf("phentsize: %04x\n", header->phentsize);
    printf("phnum:     %04x\n", header->phnum);
    printf("shentsize: %04x\n", header->shentsize);
    printf("shnum:     %04x\n", header->shnum);
    printf("shstrndx:  %04x\n", header->shstrndx);
    printf("\n");
}

/**
 * Local utility function to print the ELF section header
 * @param header Pointer to the ELF section header to proint
 */
static void printElfSectionHeader(BFIELFSectionHeader *header) {
    printf("nameIdx:   %08x\n", header->nameIdx);
    printf("type:      %08x\n", header->type);
    printf("flags:     %08x\n", header->flags);
    printf("addr:      %08x\n", header->addr);
    printf("offset:    %08x\n", header->offset);
    printf("size:      %08x\n", header->size);
    printf("link:      %08x\n", header->link);
    printf("info:      %08x\n", header->info);
    printf("addralign: %08x\n", header->addralign);
    printf("entsize:   %08x\n", header->entsize);
}



std::vector<uint8_t> BFIPatcher::patchBinary(std::vector<uint8_t> newBinary) {
    int i, textCount = 0;
    
    this->OpenCLBinary = newBinary;
    bfi_printf("this->unpatchedBinary.size(): %d\n", this->OpenCLBinary.size());
    this->locateInner();
    this->readELFSections();
    
    for (i = 0; i < this->ElfSections.size(); i++) {
        if (this->ElfSections[i].name == ".text") {
            bfi_printf("Found .text section %d size %d offset %d\n", i, this->ElfSections[i].size, this->ElfSections[i].offset);
            textCount++;
            if (textCount == 2) {
                // Patch the second .text section
                this->patchOpcodes(this->ElfSections[i].offset, this->ElfSections[i].size);
            }
        }
    }
    if (textCount != 2) {
        printf("Fatal error: Cannot find 2 .text sections!\n");
        exit(1);
    }
    
    return this->OpenCLBinary;
}

uint64_t BFIPatcher::locateInner() {
    // ELF prefix tag - 0x7f ELF
    uint8_t matchValues[] = {0x7f, 'E', 'L', 'F'};
    uint64_t index;
    // Count for found ELFs - should only find one inner ELF.
    uint8_t elfFound = 0;
    
    // We want to skip the outer ELF at offset 0 - start from offset 1.
    for (index = 1; index < this->OpenCLBinary.size() - 4; index++) {
        // Slide through and check each offset.  If it matches, we have a header!
        if (memcmp(&this->OpenCLBinary[index], matchValues, 4) == 0) {
            if (!elfFound) {
                // ELF found - report it.
                elfFound = 1;
                this->innerELFOffset = index;
                bfi_printf("Found inner ELF at offset %u\n", index);
            } else {
                // Another ELF found - fatal error.
                printf("Fatal error: Second ELF found at offset %u\n", index);
                exit(1);
            }
        }
    }
}


void BFIPatcher::readELFSections() {
    BFIELFHeader *header;
    BFIELFSectionHeader *sectionHeader;
    BFIELFSectionHeader *rawSectionHeader;
    uint64_t sectionStartOffset, sectionNameOffset;
    uint32_t sectionId;
    
    SectionInfo SectionInformation;
    
    
    header = (BFIELFHeader *) &this->OpenCLBinary[this->innerELFOffset];
    
#if BFI_DEBUG
    bfi_printf("Reading ELF header...\n");
    printElfHeader(header);
#endif
    
    // Check the first 8 bytes - if this is not correct, we have not found an ELF.
    // A gnome, perhaps.
    if (header->ident1 != 0x64010101464c457f) {
        printf("Fatal error: ELF ident1 incorrect!\n");
        exit(1);
    }
    
    // This is being checked in the Python so I check it here!
    if (header->shoff == 0) {
        printf("Fatal error: ELF shoff = 0!\n");
        exit(1);
    }
    
    // If the section header entry size does not match our struct, something is
    // quite wrong and we should exit.
    if (header->shentsize != sizeof(BFIELFSectionHeader)) {
        printf("Fatal error: Section Header Entry Size mismatch!\n");
        exit(1);
    }
    
    // Read the first section header for the text offset.
    sectionHeader = (BFIELFSectionHeader *) &this->OpenCLBinary[
            this->innerELFOffset + (header->shoff + (header->shstrndx * header->shentsize))];
#if BFI_DEBUG
    printElfSectionHeader(sectionHeader);
#endif    
    // Get the section start offset.
    sectionStartOffset = this->innerELFOffset + header->shoff;
    
    for (sectionId = 0; sectionId < header->shnum; sectionId++) {
        bfi_printf("Section ID %d\n", sectionId);
        rawSectionHeader = (BFIELFSectionHeader *) &this->OpenCLBinary[
                sectionStartOffset + sectionId * header->shentsize];

#if BFI_DEBUG
        printElfSectionHeader(rawSectionHeader);
#endif

        bfi_printf("Text: %s\n\n", (char *)&this->OpenCLBinary[
                this->innerELFOffset + sectionHeader->offset + rawSectionHeader->nameIdx]);
        
        // Clear the name field.
        SectionInformation.name.clear();
        sectionNameOffset = this->innerELFOffset + sectionHeader->offset + rawSectionHeader->nameIdx;
        // Read until a null, copying the bytes to the string.
        while (this->OpenCLBinary[sectionNameOffset]) {
            SectionInformation.name += (char) this->OpenCLBinary[sectionNameOffset];
            sectionNameOffset++;
        }
        SectionInformation.offset = rawSectionHeader->offset;
        SectionInformation.size = rawSectionHeader->size;
        this->ElfSections.push_back(SectionInformation);
    }
}

void BFIPatcher::patchOpcodes(uint64_t offset, uint64_t size) {
    uint64_t *inst;
    
    uint64_t i, numPatched = 0;
    
    // Set up the pointer at the right offset so we can access as an array.
    inst = (uint64_t *)&this->OpenCLBinary[offset + this->innerELFOffset];
    
    // Loop through the instructions.
    for (i = 0; i < (size / 8); i++) {
        bfi_printf("inst[%d]: %016lx\n", i, inst[i]);
        if ((inst[i] & (uint64_t)0x9003f00002001000) == (uint64_t)0x0001a00000000000) {
            bfi_printf("Instruction %d is byte_align!\n", i);
            numPatched++;
            inst[i] ^=  (0x0001a00000000000 ^ 0x0000c00000000000);
        }
    }
    bfi_printf("Num patched: %d\n", numPatched);
}

#if UNIT_TEST
// Needed for the unit test
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char *argv[]) {

    FILE *input;
    
    struct stat filestatus;
    BFIPatcher Patcher;
    
    std::vector<uint8_t> binaryBlob;
    
    if (argc != 2) {
        printf("Usage: %s [binary blob file]\n", argv[0]);
        exit(1);
    }
    
    // Open the input file
    input = fopen(argv[1], "rb");
    
    if (input) {
        stat(argv[1], &filestatus);
        printf("Data size: %d bytes\n", filestatus.st_size);
        
        binaryBlob.resize(filestatus.st_size, 0);
        
        fread(&binaryBlob[0], filestatus.st_size, 1, input);
    } else {
        printf("Could not open file %s!\n", argv[1]);
        exit(1);
    }
    
    Patcher.patchBinary(binaryBlob);
    
}

#endif