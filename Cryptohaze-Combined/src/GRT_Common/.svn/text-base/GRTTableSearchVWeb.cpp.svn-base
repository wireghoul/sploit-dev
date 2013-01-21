#include "GRT_Common/GRTTableSearchV1.h"
#include <fcntl.h>
#include <sys/stat.h>
#include "GRT_Common/GRTTableHeaderVWeb.h"
#include "GRT_Common/GRTTableSearchVWeb.h"
#include "GRT_Common/GRTCommon.h"
#include <string.h>
#include <vector>
#include <algorithm>
#include <errno.h>
#include <curl/curl.h>


extern char silent;

#include "CH_Common/Timer.h"

#define UNIT_TEST 0

#define MAX_HASHES_PER_REQUEST 50000

// Handle a read pf the table header
size_t table_search_write(void *buffer, size_t size, size_t nmemb, void *userp) {

    //char *foo = (char *)buffer;
    //foo[size * nmemb] = 0;
    //printf("RESULTS: %s", foo);

    // Get us a vector pointer.
    std::vector<uint8_t> *returnBuffer = (std::vector<uint8_t> *)userp;
    uint8_t *bufferPointer = (uint8_t *)buffer;

    // Allocate as much space as we need
    returnBuffer->reserve(returnBuffer->size() + (size * nmemb));

    for (int i = 0; i < (size * nmemb); i++) {
        returnBuffer->push_back(bufferPointer[i]);
    }
    //printf("Size received: %d\n", size * nmemb);

    return (size * nmemb);
}


// Handle a write from the "is table valid" bit.
void parseChainsToRegen(std::vector<uint8_t> *returnedData, std::vector<hashPasswordData> * chainsToRegen) {
    hashPasswordData regenChain;

    //printf("table_search_write, size %d\n", nmemb * size);

    uint32_t bytesCopied = 0;
    uint32_t currentPasswordLength = 0;

    // Clear out the chain to regen
    memset(&regenChain, 0, sizeof(hashPasswordData));

    // Copy data from the buffer into the chains to regen
    while (bytesCopied < returnedData->size()) {
        // Iterate through all bytes

        // If we have a newline, submit what we have.
        if (returnedData->at(bytesCopied) == '\n') {
            if (strlen((char *)regenChain.password)) {
                //printf("Pushing password %s\n", regenChain.password);
                chainsToRegen->push_back(regenChain);
                currentPasswordLength = 0;
                memset(&regenChain, 0, sizeof(hashPasswordData));
            }
            bytesCopied++;
            continue;
        }
        if (currentPasswordLength >= MAX_PASSWORD_LENGTH) {
            //printf("ERROR: Exceeding max password length... %d\n", currentPasswordLength);
            // Should not happen - return 0.
            return;
        }
        regenChain.password[currentPasswordLength] = returnedData->at(bytesCopied);
        bytesCopied++;
        currentPasswordLength++;
    }

    //printf("Return values:\n");
    //printf("%s", buffer);
    return;
}


GRTTableSearchVWeb::GRTTableSearchVWeb() {
    this->Table_Header = NULL;

    this->bitsInHash = 0;
    this->bitsInPassword = 0;

    this->CrackDisplay = NULL;
}

GRTTableSearchVWeb::~GRTTableSearchVWeb() {
    this->chainsToRegen.clear();
}

// Sets the table filename to search.
void GRTTableSearchVWeb::SetTableFilename(const char *newTableFilename) {
    this->tableFilename = newTableFilename;
}

// Give the table searcher the list of hashes it needs to find.
void GRTTableSearchVWeb::SetCandidateHashes(std::vector<hashData>* newCandidateHashes) {
    this->candidateHashes = newCandidateHashes;
}

// Actually searches the table.
void GRTTableSearchVWeb::SearchTable() {
    // Do the web search stuff...

    // Working on the binary type

#define BINARY_POST_TYPE_CONDENSED

#ifdef ASCII_HEX_POST_TYPE
    // Create the list of hashes to submit.
    char hashAscii[256];
    std::string hashStringToSubmit;
    uint32_t candidateIndexStart;
    uint32_t endHashForSegment;
    CURL *curl;
    CURLcode res;

    Timer elapsedTime;
    double searchRate;

    Timer segmentSearchTime;
    double segmentTime;
    
    // How many hashes to search for per request.
    // Tune this to get optimum performance.
    int hashesPerRequest = 10000;

    // Target time: 5s
    double targetSegmentTime = 5.0;
    
    // Buffer to handle the returned data before we parse it.
    std::vector <uint8_t> returnBuffer;

    elapsedTime.start();


    // Loop 1000 at a time to update the display
    for (candidateIndexStart = 0;
            candidateIndexStart < this->candidateHashes->size();
            candidateIndexStart += hashesPerRequest) {

        segmentSearchTime.start();

        // Build the POST query string.
        hashStringToSubmit = "candidateHashesFilename=";
        hashStringToSubmit += this->tableFilename;
        hashStringToSubmit += "&";
        hashStringToSubmit += "candidateHashes=";

        // Calculate the last hash to submit.
        endHashForSegment = candidateIndexStart + hashesPerRequest;
        if (endHashForSegment > this->candidateHashes->size()) {
            endHashForSegment = this->candidateHashes->size();
        }

        // Copy in all the hashes for this segment.
        for (int i = candidateIndexStart; i < endHashForSegment; i++) {
            memset(hashAscii, 0, 256);
            for(int j = 0; j < 16; j++) {
                sprintf(hashAscii, "%s%02x", hashAscii, this->candidateHashes->at(i).hash[j]);
            }
            hashStringToSubmit += hashAscii;
            hashStringToSubmit += "|";
        }


        curl = curl_easy_init();
        if (curl) {
            curl_easy_setopt(curl, CURLOPT_URL, this->tableURL.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, hashStringToSubmit.c_str());
            // Pass a pointer to our tableValid variable for the callback.
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &returnBuffer);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, table_search_write);

            // If we have a username, set username/password/authentication.
            if (this->tableUsername.length()) {
                curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
                curl_easy_setopt(curl, CURLOPT_USERNAME, this->tableUsername.c_str());
                curl_easy_setopt(curl, CURLOPT_PASSWORD, this->tablePassword.c_str());
            }

            res = curl_easy_perform(curl);

            if (res != CURLE_OK) {
                // Error: Something is wrong.
                printf("curl error: %s\n", curl_easy_strerror(res));
                exit(1);
            }
            /* always cleanup */
            curl_easy_cleanup(curl);
        }

        // Copy the data out of the buffer into our CH buffer
        parseChainsToRegen(&returnBuffer, &this->chainsToRegen);
        returnBuffer.clear();

        // Done with the search/parse: Check our time.
        segmentTime = segmentSearchTime.stop();

        // Update the display
        if (this->CrackDisplay) {
            // Set the percentage done
            this->CrackDisplay->setStagePercent(100.0 *
                    (float)endHashForSegment / (float)this->candidateHashes->size());

            //searchRate = (double)endHashForSegment / elapsedTime.stop();
            searchRate = (double)hashesPerRequest / segmentTime;
            this->CrackDisplay->setThreadCrackSpeed(0, CPU_THREAD, searchRate);

        } else {
            printf("\rStep %d / %d (%0.2f%%)        ",
                endHashForSegment, this->candidateHashes->size(),
                (100.0 * (float)endHashForSegment / (float)this->candidateHashes->size()));
            fflush(stdout);
        }

        // This does not work properly with the for loop.
        //hashesPerRequest = hashesPerRequest * (targetSegmentTime / segmentTime);
    }
#endif

#ifdef BINARY_POST_TYPE
    uint32_t candidateIndexStart;
    uint32_t endHashForSegment;
    CURL *curl;
    CURLcode res;
    // structures to build the post
    struct curl_httppost* post = NULL;
    struct curl_httppost* last = NULL;

    Timer elapsedTime;
    double searchRate;

    Timer segmentSearchTime;
    double segmentTime;

    // How many hashes to search for per request.
    // Tune this to get optimum performance.
    int hashesPerRequest = 10000;

    // Target time: 5s
    double targetSegmentTime = 5.0;

    // Buffer to handle the returned data before we parse it.
    std::vector <uint8_t> returnBuffer;

    // Vector to store the string of hashes we are submitting.
    std::vector <uint8_t> hashVector;

    // For now, everything is 16 bytes...
    char candidateHashLengthBytes[5] = "16";
    char hashesPerRequestBuffer[1000];

    elapsedTime.start();

    // Loop 1000 at a time to update the display
    for (candidateIndexStart = 0;
            candidateIndexStart < this->candidateHashes->size();
            candidateIndexStart += hashesPerRequest) {

        segmentSearchTime.start();

        // Calculate the last hash to submit.
        endHashForSegment = candidateIndexStart + hashesPerRequest;
        if (endHashForSegment > this->candidateHashes->size()) {
            endHashForSegment = this->candidateHashes->size();
        }

        // For handling the case of the end hash
        //hashesPerRequest = endHashForSegment - candidateIndexStart;
        sprintf(hashesPerRequestBuffer, "%d", endHashForSegment - candidateIndexStart);

        // Pre-allocate space to speed things up.
        hashVector.clear();
        hashVector.reserve((endHashForSegment - candidateIndexStart) * 16);

        // Copy in all the hashes for this segment.
        for (int i = candidateIndexStart; i < endHashForSegment; i++) {
            for(int j = 0; j < 16; j++) {
                hashVector.push_back(this->candidateHashes->at(i).hash[j]);
            }
        }

        curl = curl_easy_init();
        if (curl) {

            // Add the elements to the form.
            curl_formadd(&post, &last, CURLFORM_COPYNAME, "rawCandidateHashes",
                CURLFORM_COPYCONTENTS, "1", CURLFORM_END);
            curl_formadd(&post, &last, CURLFORM_COPYNAME, "rawHashLength",
                CURLFORM_PTRCONTENTS, candidateHashLengthBytes, CURLFORM_END);
            curl_formadd(&post, &last, CURLFORM_COPYNAME, "hashesInRequest",
                CURLFORM_PTRCONTENTS, hashesPerRequestBuffer, CURLFORM_END);
            curl_formadd(&post, &last, CURLFORM_COPYNAME, "candidateHashesFilename",
                CURLFORM_COPYCONTENTS, this->tableFilename.c_str(), CURLFORM_END);
            curl_formadd(&post, &last,
                CURLFORM_COPYNAME, "rawCandidateHashData",
                CURLFORM_BUFFER, "rawCandidateHashData",
                CURLFORM_BUFFERPTR, &hashVector[0],
                CURLFORM_BUFFERLENGTH, (long)hashVector.size(),
                CURLFORM_END);



            curl_easy_setopt(curl, CURLOPT_URL, this->tableURL.c_str());
            // Add the post data
            curl_easy_setopt(curl, CURLOPT_HTTPPOST, post);
            // Pass a pointer to our tableValid variable for the callback.
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &returnBuffer);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, table_search_write);

            // If we have a username, set username/password/authentication.
            if (this->tableUsername.length()) {
                curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
                curl_easy_setopt(curl, CURLOPT_USERNAME, this->tableUsername.c_str());
                curl_easy_setopt(curl, CURLOPT_PASSWORD, this->tablePassword.c_str());
            }

            res = curl_easy_perform(curl);

            if (res != CURLE_OK) {
                // Error: Something is wrong.
                printf("curl error: %s\n", curl_easy_strerror(res));
                exit(1);
            }
            /* always cleanup */
            curl_easy_cleanup(curl);
            curl_formfree(post);
            post = NULL;
            last = NULL;
        }

        // Copy the data out of the buffer into our CH buffer
        parseChainsToRegen(&returnBuffer, &this->chainsToRegen);
        returnBuffer.clear();

        // Done with the search/parse: Check our time.
        segmentTime = segmentSearchTime.stop();

        // Update the display
        if (this->CrackDisplay) {
            // Set the percentage done
            this->CrackDisplay->setStagePercent(100.0 *
                    (float)endHashForSegment / (float)this->candidateHashes->size());

            //searchRate = (double)endHashForSegment / elapsedTime.stop();
            searchRate = (double)hashesPerRequest / segmentTime;
            this->CrackDisplay->setThreadCrackSpeed(0, CPU_THREAD, searchRate);

        } else {
            printf("\rStep %d / %d (%0.2f%%)        ",
                endHashForSegment, this->candidateHashes->size(),
                (100.0 * (float)endHashForSegment / (float)this->candidateHashes->size()));
            fflush(stdout);
        }

        //hashesPerRequest = hashesPerRequest * (targetSegmentTime / segmentTime);
    }
#endif


#ifdef BINARY_POST_TYPE_CONDENSED
    uint32_t candidateIndexStart;
    uint32_t endHashForSegment;
    CURL *curl;
    CURLcode res;
    // structures to build the post
    struct curl_httppost* post = NULL;
    struct curl_httppost* last = NULL;

    Timer elapsedTime;
    double searchRate;

    Timer segmentSearchTime;
    double segmentTime;

    // How many hashes to search for per request.
    // Tune this to get optimum performance.
    int hashesPerRequest = 1000;

    // Target time: 15s
    double targetSegmentTime = 15.0;

    // Buffer to handle the returned data before we parse it.
    std::vector <uint8_t> returnBuffer;

    // Vector to store the string of hashes we are submitting.
    std::vector <uint8_t> hashVector;

    // For now, everything is 16 bytes...
    char candidateHashLengthBytes[5];;
    char hashesPerRequestBuffer[1000];

    // Figure out how many bytes to copy
    int significantBytesToCopy = 0;


    if ((this->Table_Header->getBitsInHash() % 8) == 0) {
        // If the number of bits is an even byte, we're good!
        significantBytesToCopy = this->Table_Header->getBitsInHash() / 8;
    } else {
        // Else, we're not so good, and need to round up.
        significantBytesToCopy = this->Table_Header->getBitsInHash() / 8;
        significantBytesToCopy++;
    }

    //printf("Bits in hash: %d\n", this->Table_Header->getBitsInHash());
    //printf("Bytes to send: %d\n", significantBytesToCopy);

    sprintf(candidateHashLengthBytes, "%d", significantBytesToCopy);

    elapsedTime.start();

    // Loop 1000 at a time to update the display
    /*for (candidateIndexStart = 0;
            candidateIndexStart < this->candidateHashes->size();
            candidateIndexStart += hashesPerRequest) {*/
    candidateIndexStart = 0;
    while (candidateIndexStart < this->candidateHashes->size()) {

        segmentSearchTime.start();

        // Calculate the last hash to submit.
        endHashForSegment = candidateIndexStart + hashesPerRequest;
        if (endHashForSegment > this->candidateHashes->size()) {
            endHashForSegment = this->candidateHashes->size();
        }

        // For handling the case of the end hash
        //hashesPerRequest = endHashForSegment - candidateIndexStart;
        sprintf(hashesPerRequestBuffer, "%d", endHashForSegment - candidateIndexStart);

        // Pre-allocate space to speed things up.
        hashVector.clear();
        hashVector.reserve((endHashForSegment - candidateIndexStart) * significantBytesToCopy);

        // Copy in all the hashes for this segment.
        for (int i = candidateIndexStart; i < endHashForSegment; i++) {
            for(int j = 0; j < significantBytesToCopy; j++) {
                hashVector.push_back(this->candidateHashes->at(i).hash[j]);
            }
        }

        curl = curl_easy_init();
        if (curl) {

            // Add the elements to the form.
            curl_formadd(&post, &last, CURLFORM_COPYNAME, "rawCandidateHashesCondensed",
                CURLFORM_COPYCONTENTS, "1", CURLFORM_END);
            curl_formadd(&post, &last, CURLFORM_COPYNAME, "rawHashLength",
                CURLFORM_PTRCONTENTS, candidateHashLengthBytes, CURLFORM_END);
            curl_formadd(&post, &last, CURLFORM_COPYNAME, "hashesInRequest",
                CURLFORM_PTRCONTENTS, hashesPerRequestBuffer, CURLFORM_END);
            curl_formadd(&post, &last, CURLFORM_COPYNAME, "candidateHashesFilename",
                CURLFORM_COPYCONTENTS, this->tableFilename.c_str(), CURLFORM_END);
            curl_formadd(&post, &last,
                CURLFORM_COPYNAME, "rawCandidateHashData",
                CURLFORM_BUFFER, "rawCandidateHashData",
                CURLFORM_BUFFERPTR, &hashVector[0],
                CURLFORM_BUFFERLENGTH, (long)hashVector.size(),
                CURLFORM_END);



            curl_easy_setopt(curl, CURLOPT_URL, this->tableURL.c_str());
            // Add the post data
            curl_easy_setopt(curl, CURLOPT_HTTPPOST, post);
            // Pass a pointer to our tableValid variable for the callback.
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &returnBuffer);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, table_search_write);

            // If we have a username, set username/password/authentication.
            if (this->tableUsername.length()) {
                curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
                curl_easy_setopt(curl, CURLOPT_USERNAME, this->tableUsername.c_str());
                curl_easy_setopt(curl, CURLOPT_PASSWORD, this->tablePassword.c_str());
            }

            res = curl_easy_perform(curl);

            if (res != CURLE_OK) {
                // Error: Something is wrong.
                printf("curl error: %s\n", curl_easy_strerror(res));
                exit(1);
            }
            /* always cleanup */
            curl_easy_cleanup(curl);
            curl_formfree(post);
            post = NULL;
            last = NULL;
        }

        // Copy the data out of the buffer into our CH buffer
        parseChainsToRegen(&returnBuffer, &this->chainsToRegen);
        returnBuffer.clear();

        // Done with the search/parse: Check our time.
        segmentTime = segmentSearchTime.stop();

        // Update the display
        if (this->CrackDisplay) {
            // Set the percentage done
            this->CrackDisplay->setStagePercent(100.0 *
                    (float)endHashForSegment / (float)this->candidateHashes->size());

            //searchRate = (double)endHashForSegment / elapsedTime.stop();
            searchRate = (double)hashesPerRequest / segmentTime;
            this->CrackDisplay->setThreadCrackSpeed(0, CPU_THREAD, searchRate);

        } else {
            printf("\rStep %d / %d (%0.2f%%)        ",
                endHashForSegment, this->candidateHashes->size(),
                (100.0 * (float)endHashForSegment / (float)this->candidateHashes->size()));
            fflush(stdout);
        }
        candidateIndexStart += hashesPerRequest;
        hashesPerRequest = hashesPerRequest * (targetSegmentTime / segmentTime);
        // Prevent oversize requests.
        if (hashesPerRequest > MAX_HASHES_PER_REQUEST) {
            hashesPerRequest = MAX_HASHES_PER_REQUEST;
        }
    }
#endif

}

// Return a list of the chains to regenerate
// This should probably get deleted after use :)
std::vector<hashPasswordData>* GRTTableSearchVWeb::getChainsToRegen() {
    std::vector<hashPasswordData>* returnChains;

    returnChains = new std::vector<hashPasswordData>();

    *returnChains = this->chainsToRegen;

    return returnChains;
}

void GRTTableSearchVWeb::setTableHeader(GRTTableHeader * newTableHeader) {
    this->Table_Header = (GRTTableHeaderVWeb *)newTableHeader;

    this->bitsInPassword = this->Table_Header->getBitsInPassword();
    this->bitsInHash = this->Table_Header->getBitsInHash();

}

void GRTTableSearchVWeb::getChainAtIndex(uint64_t index, struct hashPasswordData *chainInfo) {

}

uint64_t GRTTableSearchVWeb::getNumberChains() {
    return 0;
}


int GRTTableSearchVWeb::getBitsInHash() {
    return this->bitsInHash;
}

int GRTTableSearchVWeb::getBitsInPassword() {
    return this->bitsInPassword;
}
void GRTTableSearchVWeb::setCrackDisplay(GRTCrackDisplay* newCrackDisplay) {
    this->CrackDisplay = newCrackDisplay;
}

#if UNIT_TEST

int main() {
    GRTTableSearchVWeb *TableSearch;
    GRTTableHeaderVWeb *TableHeader;

    TableHeader = new GRTTableHeaderVWeb();
    TableSearch = new GRTTableSearchVWeb();

    if (TableHeader->isValidTable("NTLM-len6-idx0-chr10-cl5000-v2-perfect.grt", 0)) {
        printf("Table is valid!\n");
    } else {
        printf("Table is NOT valid!\n");
        exit;
    }

    TableHeader->readTableHeader("NTLM-len6-idx0-chr10-cl5000-v2-perfect.grt");


    TableHeader->printTableHeader();

    TableSearch->SetTableFilename("NTLM-len6-idx0-chr10-cl5000-v2-perfect.grt");
    TableSearch->setTableHeader(TableHeader);

    // Gen us some candidate hashes

    std::vector<hashData> candidateHashes;
    hashData hash;
    std::vector<hashPasswordData> *chainsToRegen;

    memset(&hash, 0, sizeof(hashData));
    
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 16; j++) {
            hash.hash[j] = i;
        }
        candidateHashes.push_back(hash);
    }

    TableSearch->SetCandidateHashes(&candidateHashes);

    TableSearch->SearchTable();

    chainsToRegen = TableSearch->getChainsToRegen();

    printf("Got %d chains back\n", chainsToRegen->size());

}

#endif
