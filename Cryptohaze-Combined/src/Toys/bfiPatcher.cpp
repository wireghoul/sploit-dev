// Try out BFI_INT patching more correctly given a binary.

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <stdint.h>
#include <sys/stat.h>

#ifdef WIN32
#include <stdint.h>
#else
#include <unistd.h>
#endif

#include <string.h>
#include <string>


// This class is a C++ copy of http://svn3.xp-dev.com/svn/phoenix-miner/trunk/kernels/phatk/BFIPatcher.py


#pragma pack(push)
#pragma pack(1)
typedef struct ELFHeader {
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
} ELFHeader;

typedef struct ELFSectionHeader {
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
} ELFSectionHeader;
#pragma pack(pop)

typedef struct SectionInfo {
    std::string name;
    uint64_t offset;
    uint64_t size;
} SectionInfo;

static void printElfHeader(ELFHeader *header) {
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

static void printElfSectionHeader(ELFSectionHeader *header) {
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

class BFIPatcher {
public:
    std::vector<uint8_t> patchBinary(std::vector<uint8_t>);
private:
    // The unpatched blob
    std::vector<uint8_t> unpatchedBinary;
    
    std::vector<SectionInfo> ElfSections;
    
    // Inner ELF offset
    uint64_t innerELFOffset;
    
    // Locate the inner ELF.
    uint64_t locateInner();
    
    // read in the ELF sections
    void readELFSections();
    
    void patchOpcodes(uint64_t offset, uint64_t size);
    
};

std::vector<uint8_t> BFIPatcher::patchBinary(std::vector<uint8_t> newBinary) {
    int i, textCount = 0;
    
    this->unpatchedBinary = newBinary;
    printf("this->unpatchedBinary.size(): %d\n", this->unpatchedBinary.size());
    this->locateInner();
    this->readELFSections();
    
    for (i = 0; i < this->ElfSections.size(); i++) {
        if (this->ElfSections[i].name == ".text") {
            printf("Found .text section %d size %d offset %d\n", i, this->ElfSections[i].size, this->ElfSections[i].offset);
            textCount++;
            if (textCount == 2) {
                // Patch the second .text section
                this->patchOpcodes(this->ElfSections[i].offset, this->ElfSections[i].size);
            }
        }
    }
    if (textCount != 2) {
        printf("Fatal error: Cannot find 2 .text sections!\n");
    }
    
    return this->unpatchedBinary;
}

uint64_t BFIPatcher::locateInner() {
    // Find the offset of the second ELF.
    // This is 0x7f . E . L . F
    
    uint8_t matchValues[] = {0x7f, 'E', 'L', 'F'};
    uint64_t index;
    uint8_t elfFound = 0;
    
    for (index = 1; index < this->unpatchedBinary.size() - 4; index++) {
        if (memcmp(&this->unpatchedBinary[index], matchValues, 4) == 0) {
            
            if (!elfFound) {
                // ELF found - report it.
                elfFound = 1;
                this->innerELFOffset = index;
                printf("Found inner ELF at offset %u\n", index);
            } else {
                // Another ELF found - fatal error.
                printf("Fatal error: Second ELF found at offset %u\n", index);
                exit(1);
            }
        }
    }
 }

void BFIPatcher::readELFSections() {
    ELFHeader *header;
    ELFSectionHeader *sectionHeader;
    ELFSectionHeader *rawSectionHeader;
    uint64_t sectionStartOffset, sectionNameOffset;
    uint32_t sectionId;
    
    SectionInfo SectionInformation;
    
    printf("Reading ELF header...\n");
    
    header = (ELFHeader *) &this->unpatchedBinary[this->innerELFOffset];
    printElfHeader(header);
    
    if (header->ident1 != 0x64010101464c457f) {
        printf("Fatal error: ELF ident1 invalid!\n");
        exit(1);
    }
    
    if (header->shoff == 0) {
        printf("Fatal error: ELF shoff = 0!\n");
        exit(1);
    }
    
    if (header->shentsize != sizeof(ELFSectionHeader)) {
        printf("Fatal error: Section Header Entry Size mismatch!\n");
        exit(1);
    }
    
    sectionHeader = (ELFSectionHeader *) &this->unpatchedBinary[
            this->innerELFOffset + (header->shoff + (header->shstrndx * header->shentsize))];
    printElfSectionHeader(sectionHeader);
    
    // Get the section start offset.
    sectionStartOffset = this->innerELFOffset + header->shoff;
    
    for (sectionId = 0; sectionId < header->shnum; sectionId++) {
        printf("Section ID %d\n", sectionId);
        rawSectionHeader = (ELFSectionHeader *) &this->unpatchedBinary[sectionStartOffset + sectionId * header->shentsize];
        printElfSectionHeader(rawSectionHeader);
        printf("Text: %s\n\n", (char *)&this->unpatchedBinary[this->innerELFOffset + sectionHeader->offset + rawSectionHeader->nameIdx]);
        SectionInformation.name.clear();
        sectionNameOffset = this->innerELFOffset + sectionHeader->offset + rawSectionHeader->nameIdx;
        while (this->unpatchedBinary[sectionNameOffset]) {
            SectionInformation.name += (char) this->unpatchedBinary[sectionNameOffset];
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
    
    inst = (uint64_t *)&this->unpatchedBinary[offset + this->innerELFOffset];
    
    // Loop through the instructions.
    for (i = 0; i < (size / 8); i++) {
        printf("inst[%d]: %016lx\n", i, inst[i]);
        if ((inst[i] & (uint64_t)0x9003f00002001000) == (uint64_t)0x0001a00000000000) {
            printf("Instruction %d is byte_align!\n", i);
            numPatched++;
            inst[i] ^=  (0x0001a00000000000 ^ 0x0000c00000000000);
        }
    }
    printf("Num patched: %d\n", numPatched);
}

/*

int main() {

    FILE *input;
    
    struct stat filestatus;
    BFIPatcher Patcher;
    
    std::vector<uint8_t> binaryBlob;
    
    // Open the input file
    input = fopen("amd.bin", "rb");
    
    if (input) {
        stat("amd.bin", &filestatus);
        printf("Data size: %d bytes\n", filestatus.st_size);
        
        binaryBlob.resize(filestatus.st_size, 0);
        
        fread(&binaryBlob[0], filestatus.st_size, 1, input);
    } else {
        printf("Could not open file!\n");
        exit(1);
    }
    
    Patcher.patchBinary(binaryBlob);
    
}
*/