#include <fcntl.h>
#include <sys/stat.h>
#include "GRT_Common/GRTTableHeaderV3.h"
#include "GRT_Common/GRTTableSearchV3.h"
#include "GRT_Common/GRTCommon.h"
#include "CH_Common/CHRandom.h"
#include <string.h>
#include <vector>
#include <algorithm>
#include <errno.h>
#include <string>

extern char silent;

// Fix for 64-bit stat on Windows platform
// Thanks MaddGamer for finding this problem
// with big files!
#if _MSC_VER >= 1000
#define stat    _stat64
#endif

GRTTableSearchV3::GRTTableSearchV3() {
    this->maxChainAddress = 0;
    this->tableSize = 0;
    this->tableFile = 0;
    this->tableAccess = NULL;
    this->tableMemoryBase = NULL;
    this->Table_Header = NULL;

    this->outputFile = NULL;
    this->numberChainsWritten = 0;

    this->bitsInHash = 0;
    this->bitsInPassword = 0;
}

GRTTableSearchV3::~GRTTableSearchV3() {

}

// Sets the table filename to search.
void GRTTableSearchV3::SetTableFilename(const char *newTableFilename) {
    struct stat file_status;
    uint64_t i;
    uint32_t passwordPosition, randomValue;
    CHRandom RandomGenerator;

    if(stat(newTableFilename, &file_status) != 0){
        printf("Unable to stat %s\n", newTableFilename);
        exit(1);
    }

    // If we do not have a table header, read it.
    if (!this->Table_Header) {
        this->Table_Header = new GRTTableHeaderV3();
        this->Table_Header->readTableHeader(newTableFilename);
        this->bitsInHash = this->Table_Header->getBitsInHash();
        this->bitsInPassword = this->Table_Header->getBitsInPassword();
        this->bytesInChain = (this->bitsInHash + this->bitsInPassword) / 8;
    }

    //this->Table_Header->printTableHeader();

#if USE_BOOST_MMAP
    // Use boost memory mapped files to get the table loaded.
    mapped_file_params table_params;

    table_params.path = newTableFilename;

    this->boost_mapped_source.open(table_params);
    this->tableMemoryBase = (unsigned char *)this->boost_mapped_source.data();
#else
    this->tableFile = open(newTableFilename, O_RDONLY);
    if (!this->tableFile) {
        printf("Cannot open table file %s!\n", newTableFilename);
        exit(1);
    }

    // Memory map the file.  This starts at the BASE - we deal with the offsets later.
    // Windows has some weird issues with offsets.  So we deal with that here.
    this->tableMemoryBase = (unsigned char *)mmap(0, file_status.st_size, PROT_READ, MAP_SHARED, this->tableFile, 0);

    // Advise the OS that access will be sequential.
    madvise (this->tableMemoryBase, file_status.st_size, MADV_SEQUENTIAL);
#endif

    // Copy the table size over.
    this->tableSize = file_status.st_size;

    // Now do our offset work.  Table header v1 is 8192 bytes by definition.
    this->tableAccess = (unsigned char *) (this->tableMemoryBase + 8192);

    this->maxChainAddress = (this->tableSize - 8192) /
            ((this->Table_Header->getBitsInHash() + this->Table_Header->getBitsInPassword()) / 8);

    // Now we have to generate the matching initial passwords for this mess.

    // Seed mt with the right seed
    //mt_seed32new(this->Table_Header->getRandomSeedValue());
    RandomGenerator.setSeed(this->Table_Header->getRandomSeedValue());

    // Get the right number of random data to get to the right offset
    for (i = 0; i < this->Table_Header->getChainStartOffset(); i++) {
        for (passwordPosition = 0; passwordPosition < this->Table_Header->getPasswordLength(); passwordPosition++) {
            //randomValue = mt_lrand();
            RandomGenerator.getRandomValue();
        }
    }

    // Theoretically, the random seed is good to go.
    char** hostCharset2D; // The 16x256 array of characters
    char *CharsetLengths;
    int charsetLength;
    char charset[512];

    memset(charset, 0, 512);

    hostCharset2D = this->Table_Header->getCharset();
    CharsetLengths = this->Table_Header->getCharsetLengths();

    charsetLength = CharsetLengths[0];

    //printf("Charset length: %d\n", charsetLength);

    for (i = 0; i < 512; i++) {
        charset[i] = hostCharset2D[0][i % charsetLength];
    }

    for (i = 0; i < 16; i++) {
        delete[] hostCharset2D[i];
    }
    delete[] hostCharset2D;
    delete[] CharsetLengths;

    // Should have the charset AND the data ready!

    std::string newPassword;
    this->initialPasswords.reserve(this->maxChainAddress);
    
    for (i = 0; i < this->maxChainAddress; i++) {
        newPassword.clear();
        for (passwordPosition = 0; passwordPosition < this->Table_Header->getPasswordLength(); passwordPosition++) {
            //newPassword += charset[mt_lrand() % charsetLength];
            newPassword += charset[RandomGenerator.getRandomValue() % charsetLength];
        }
        //printf("Password %d: %s\n", i, newPassword.c_str());
        this->initialPasswords.push_back(newPassword);
    }



}

void GRTTableSearchV3::setTableHeader(GRTTableHeader * newTableHeader) {
    this->Table_Header = (GRTTableHeaderV3 *)newTableHeader;

    this->bitsInPassword = this->Table_Header->getBitsInPassword();
    this->bitsInHash = this->Table_Header->getBitsInHash();
    this->bytesInChain = (this->bitsInHash + this->bitsInPassword) / 8;


    if ((this->bitsInHash + this->bitsInPassword) % 8) {
        printf("ERROR: Must set bitsInHash + bitsInPassword to a multiple of 8 (byte aligned)!\n");
        printf("bitsInhash: %d\n", this->bitsInHash);
        printf("bitsInPassword: %d\n", this->bitsInPassword);

        // Be useful.
        int possibleNumberOfBits = 0;
        // Calculate lower byte count
        possibleNumberOfBits = (((this->bitsInHash + this->bitsInPassword) / 8) * 8) - this->bitsInPassword;
        printf("Try %d or %d for bits in hash.\n", possibleNumberOfBits, possibleNumberOfBits + 8);

        exit(1);
    }

    //printf("Set bits in pw to %d\n", this->bitsInPassword);
    //printf("hash mask: %08x%08x\n", (uint32_t)(this->hashMask >> 32), (uint32_t)(this->hashMask & 0xffffffff));
}

void GRTTableSearchV3::getChainAtIndex(uint64_t index, struct hashPasswordData *chainInfo) {

    uint64_t addressToRead;
    uint64_t readIndex;

    // Zero the chain out before copying data in.
    memset(chainInfo, 0, sizeof(struct hashPasswordData));

    // Determine what character we start at.
    addressToRead = index * ((this->bitsInHash + this->bitsInPassword) / 8);

    // Copy the specified number of bits of hash in.
    for (readIndex = 0;
            readIndex < ((this->bitsInHash + this->bitsInPassword) / 8);
            readIndex++) {
        chainInfo->hash[readIndex] = this->tableAccess[addressToRead + readIndex];
    }
    
    // Copy the password out of the internal store.
    memcpy(chainInfo->password, this->initialPasswords.at(index).c_str(), this->Table_Header->getPasswordLength());
}

uint64_t GRTTableSearchV3::getNumberChains() {
    return this->maxChainAddress;
}


// Open a file to output data to
int GRTTableSearchV3::openOutputFile(char *outputFilename) {

    if (!this->Table_Header) {
        printf("Cannot open output file without table header!\n");
        exit(1);
    }

    this->outputFile = fopen(outputFilename, "wb");
    if (!this->outputFile) {
        printf("Cannot open output file %s!\n", outputFilename);
        exit(1);
    }
    // Write the header to the beginning of the file
    this->Table_Header->writeTableHeader(this->outputFile);
    return 1;
}
int GRTTableSearchV3::closeOutputFile() {
    // Write the new table header and close the file.
    this->Table_Header->setNumberChains(this->numberChainsWritten);
    this->Table_Header->writeTableHeader(this->outputFile);
    fclose(this->outputFile);
	return 1;
}

// Write the chain to the opened file.
// TODO: Handle byte aligned hashes/passwords
int GRTTableSearchV3::writeChain(hashPasswordData* hashToWrite) {
    if (!fwrite(hashToWrite->hash, this->bytesInChain, 1, this->outputFile)) {
        printf("ERROR WRITING DATA\n");
        return 0;
    }
    this->numberChainsWritten++;
    return 1;
}


int GRTTableSearchV3::getBitsInHash() {
    return this->bitsInHash;
}

int GRTTableSearchV3::getBitsInPassword() {
    return this->bitsInPassword;
}

