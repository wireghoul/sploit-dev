/**
 * Playing around with data conversion from "vector of vectors" to drive format.
 */


#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * Convert a vector of input data to a set of vectors suited to device access.
 * 
 * This function generically takes a vector of input vectors, and converts them
 * into a vector of data that is 32-bit coalesced for GPU access.  In other
 * words, the first 4 bytes of data0 are followed by the first 4 bytes of data1,
 * etc.  This enables coalesced access when GPUs are accessing data - if threads
 * all fetch word0, they will be accessing a contiguous region of memory.
 * 
 * This also creates a vector of 32-bit values that contain the length of each
 * element, in little endian format.  Easy access for the GPU!  This length does
 * NOT include the length of the data offset, or the padding bit - it is the
 * length of the data only.
 * 
 * Note that this function will OVERWRITE ALL DATA in the two output vectors!
 * 
 * isBigEndianData should be set if the data coming in is for a big endian hash
 * such as the SHA family, otherwise should be false.
 * 
 * addPaddingBit will append a padding bit to the end of each data element
 * before inserting it.  This is useful for salts that come after the password.
 * 
 * dataOffset is used to allow for pass.salt constructs more easily.  If this is
 * set, the data will be offset this many bytes in the first word.  This allows
 * more easily oring the salt in after the password.  This value should be in
 * the range of 0-3 or it makes no sense (or, more specifically, will be %4'd).
 * 
 * @param inputData Input data vector (vector of vectors of uint8_t)
 * @param isBigEndian Set to true for big endian hashes (SHA family, etc)
 * @param addPaddingBit Set to true to append 0x80 to all input data.
 * @param dataOffset Data offset for the first word of data.
 * @param dataLengths Output of the lengths of each element.
 * @param dataDeviceFormat Output of the coalesced data in 32-bit chunks.
 */
void convertData32(
        const std::vector<std::vector<uint8_t> > &inputData,
        char isBigEndian,
        char addPaddingBit,
        uint8_t dataOffset,
        std::vector<uint32_t> &dataLengths, 
        std::vector<uint32_t> &dataDeviceFormat
        ) {
    
    uint32_t maximumDataLength = 0; // Max data length (with padding and offset)
    uint32_t dataWordsNeeded = 0; // Number of 32-bit words to fit all data
    uint32_t dataWord, dataCount = 0;
    std::vector<std::vector<uint8_t> >::const_iterator dataIt; // Iterator for data
    
    // Clear out any existing data.
    dataLengths.clear();
    dataLengths.reserve(inputData.size());
    dataDeviceFormat.clear();
    
    // Ensure dataOffset is in the range of 0-3
    dataOffset = dataOffset % 4;
    
    /**
     * Iterate through the salts and do several things: Get the length of each
     * salt and push it into the array, add the padding bit to the end if
     * requested, and determine the maximum salt length in use for creating the
     * device length array
     */
    for (dataIt = inputData.begin(); dataIt < inputData.end(); dataIt++) {
        // Set the length - this is the *actual* length of the data, without
        // any padding or offset factored in.
        dataLengths.push_back((uint32_t)dataIt->size());

        // Determine the maximum salt length.  We include the padding bit if
        // set - it will need space in the array!
        if (dataIt->size() > maximumDataLength) {
            maximumDataLength = dataIt->size();
        }
    }
    
    // If a padding bit is used, add one to the max data length.
    if (addPaddingBit) {
        maximumDataLength++;
    }
    // Add the offset to the max data size.
    maximumDataLength += dataOffset;
    printf("Maximum salt length: %d\n", maximumDataLength);

    // Determine how many 32-bit words are needed to fit everything
    dataWordsNeeded = (maximumDataLength / 4);
    // If there are more bytes, add another word.
    if (maximumDataLength % 4) {
        dataWordsNeeded++;
    }
    printf("Max salt words: %d\n", dataWordsNeeded);
    
    // Resize the device data vector.
    dataDeviceFormat.resize(inputData.size() * dataWordsNeeded);

    for (dataIt = inputData.begin(); dataIt < inputData.end(); dataIt++) {
        dataWord = 0;
        size_t dataSize = dataIt->size();
        uint8_t dataByte;
        
        // If a padding bit is being added, increase size to account for it.
        if (addPaddingBit) {
            dataSize++;
        }

        for (uint32_t i = 0; i < dataSize; i++) {
            // Data byte is either the value from the vector, or the padding bit
            // if needed.
            if (i < dataIt->size()) {
                dataByte = dataIt->at(i);
            } else {
                dataByte = 0x80;
            }
            
            // Insert the byte into the word in the proper spot.
            if (!isBigEndian) {
                // Hash is big endian.  Byte 0 is shifted 0.
                dataWord |= (uint32_t)dataByte << (((i + dataOffset) % 4) * 8);
            } else {
                // Hash is little endian.  Byte 0 gets shifted << 24
                dataWord |= (uint32_t)dataByte << (((3 - (i + dataOffset)) % 4) * 8);
            }
            
            // Check to see if the data word needs to be pushed back.  This is
            // true if a word is full (%3 == 0), or if it is the last word in
            // the salt (== ->size() - 1).  In any case, push the word and reset
            // the storage value.
            if ((((i + dataOffset) % 4) == 3) ||
                    (i == (dataSize - 1))) {
                if (((i + dataOffset) % 4) == 3) {
                    //printf("Pushing for %%4 - i: %d  doffset: %d\n", i, dataOffset);
                }
                if (i == (dataSize - 1)) {
                    //printf("Pushing for size\n");
                }
                // Offset in the salt array: (totalSalts * word + currentSalt)
                //printf("Shoved dataword 0x%08x into position %u\n", dataWord,
                //        inputData.size() * ((i + dataOffset) / 4) + dataCount);
                dataDeviceFormat[inputData.size() *
                        ((i + dataOffset) / 4) + dataCount] = dataWord;
                dataWord = 0;
            }
        }
        dataCount++;
    }
}


int main() {
    
    std::vector<std::vector<uint8_t> > data;
    std::vector<uint32_t> dataLengths, dataDeviceFormat;
    srand ( time(NULL) );

    for (int i = 0; i < 10; i++) {
        std::vector<uint8_t> dataElement;
        for (int j = 0; j < (rand() % 10); j++) {
            dataElement.push_back(rand() & 0xff);
        }
        data.push_back(dataElement);
    }
    
    for (int i= 0; i < 10; i++) {
        printf("Element %d: ", i);
        for (int j = 0; j < data[i].size(); j++) {
            printf("%02x", data[i][j]);
        }
        printf("\n");
    }
    
    convertData32(
        data,
        0, // BE
        1, // Padding
        2, // Offset
        dataLengths, 
        dataDeviceFormat
        );
    
    for (int i = 0; i < dataDeviceFormat.size(); i++) {
        printf("%08x ", dataDeviceFormat[i]);
    }
    
}