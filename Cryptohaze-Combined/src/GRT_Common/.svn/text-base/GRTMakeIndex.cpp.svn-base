
#include "GRT_Common/GRTMakeIndexCommandLineData.h"
#include "GRT_Common/GRTTableHeaderV1.h"
#include "GRT_Common/GRTTableHeaderV2.h"
#include "GRT_Common/GRTTableSearchV1.h"
#include "GRT_Common/GRTTableSearchV2.h"


// Silence output if true.
char silent = 0;


void createTableIndex(std::string TableFilename, uint32_t BitsToIndex) {
    int i;
    FILE *indexFile;

    char filenameBuffer[2000];
    uint64_t CurrentIndex, PrevWrittenIndex, NumberWritten;

    float AverageInterval;

    uint32_t Mask, PrevWritten, CurrentMasked;

    struct indexFile indexFileData;
    struct hashPasswordData tableChain;

    GRTTableSearch *TableFile;

    int tableVersion;

    // Get the table version and create the appropriate table header
    tableVersion = getTableVersion(TableFilename.c_str());

    if (tableVersion == -1) {
        printf("ERROR making index: Cannot read table file %s!\n", TableFilename.c_str());
        return;
    }

    if (tableVersion == 1) {
        TableFile = new GRTTableSearchV1();
        TableFile->SetTableFilename(TableFilename.c_str());
    } else if (tableVersion == 2) {
        TableFile = new GRTTableSearchV2();
        TableFile->SetTableFilename(TableFilename.c_str());
        if (BitsToIndex > TableFile->getBitsInHash()) {
            printf("ERROR: Bits to index (%d) > bits in hash (%d)!\n", BitsToIndex, TableFile->getBitsInHash());
            printf("Not making this index... use fewer bits.\n");
            return;
        }
    } else {
        printf("Table version %d not supported!\n", tableVersion);
        return;
    }


    // Calculate the masks as needed now.
    Mask = 0x00000000;
    for (i = 0; i < BitsToIndex; i++) {
        // Add the needed bits.
        Mask |= (1 << (31 - i));
    }

    sprintf(filenameBuffer, "%s.idx", TableFilename.c_str());
    indexFile = fopen(filenameBuffer, "wb");

    printf("Writing to index %s\n", filenameBuffer);

    // To ensure the 0 value is written
    PrevWritten = 0xFFFFFFFF;
    PrevWrittenIndex = 0;
    AverageInterval = 0;
    NumberWritten = 0;

    for (CurrentIndex = 0; CurrentIndex < TableFile->getNumberChains(); CurrentIndex++) {
        // Get the current chain
        TableFile->getChainAtIndex(CurrentIndex, &tableChain);

        CurrentMasked = tableChain.hash[0] << 24 |
                        tableChain.hash[1] << 16 |
                        tableChain.hash[2] << 8  |
                        tableChain.hash[3];
        CurrentMasked &= Mask;

        if (CurrentIndex % 100000 == 0) {
            printf("\rCreating index... (%ld/%ld, %0.2f%% done, Interval %0.0f)    ", CurrentIndex, TableFile->getNumberChains(),
                    ((float)CurrentIndex / (float)TableFile->getNumberChains()) * 100.0,
                    AverageInterval);
            fflush(stdout);
        }

        if (CurrentMasked != PrevWritten) {

            //printf("Writing index %08X at index %d\n", CurrentMasked, CurrentIndex);
            indexFileData.Index = CurrentMasked;
            indexFileData.Offset = CurrentIndex;
            if (!fwrite(&indexFileData, sizeof(struct indexFile), 1, indexFile)) {
                printf("Error writing output!\n");
                exit(1);
            }
            NumberWritten++;

            AverageInterval = (((AverageInterval * (NumberWritten - 1)) + (CurrentIndex - PrevWrittenIndex)) / NumberWritten);

            PrevWritten = CurrentMasked;
            PrevWrittenIndex = CurrentIndex;
        }
    }

    // Done... clean up.
    fclose(indexFile);
    delete TableFile;
}




int main(int argc, char *argv[]) {

    int IndexCount;
	std::string tableFilename;

    GRTMakeIndexCommandLineData CommandLine;

    CommandLine.ParseCommandLine(argc, argv);

    for (IndexCount = 0; IndexCount < CommandLine.getNumberOfTableFiles(); IndexCount++) {
        tableFilename = CommandLine.getNextTableFile();
        printf("Creating index for file %d/%d...\n", IndexCount + 1, CommandLine.getNumberOfTableFiles());
        createTableIndex(tableFilename, CommandLine.getBitsToIndex());
    }

    printf("\n\nIndex creation complete.\n");
}