/*
Cryptohaze Multiforcer & Wordyforcer - low performance GPU password cracking
Copyright (C) 2011  Bitweasil (http://www.cryptohaze.com/)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#include "Multiforcer_Common/CHHashes.h"
#include "Multiforcer_Common/CHCommon.h"


CHHashes::CHHashes() {
    // Init hash types.
    memset(this->HashTypes, 0, sizeof(CHHashTypeData) * MAX_HASH_TYPES);
    this->CurrentHashId = -1;

    // Hash type MD5-plain
    // CHHashTypePlainMD5
    strcpy(this->HashTypes[CH_HASH_TYPE_MD5_PLAIN].HashString, "MD5");
    strcpy(this->HashTypes[CH_HASH_TYPE_MD5_PLAIN].HashDescription, "Plain unsalted MD5 hashes.");
    strcpy(this->HashTypes[CH_HASH_TYPE_MD5_PLAIN].HashAlgorithm, "md5($pass)");
    this->HashTypes[CH_HASH_TYPE_MD5_PLAIN].DefaultWorkunitSize = 32;
    this->HashTypes[CH_HASH_TYPE_MD5_PLAIN].MinSupportedLength = 1;
    this->HashTypes[CH_HASH_TYPE_MD5_PLAIN].MaxSupportedLength = 48;
    this->HashTypes[CH_HASH_TYPE_MD5_PLAIN].NetworkSupportEnabled = 1;
    this->HashTypes[CH_HASH_TYPE_MD5_PLAIN].MaxHashCount = 0;

    // Hash type MD4-plain
    // CHHashTypePlainMD4
    strcpy(this->HashTypes[CH_HASH_TYPE_MD4_PLAIN].HashString, "MD4");
    strcpy(this->HashTypes[CH_HASH_TYPE_MD4_PLAIN].HashDescription, "Plain unsalted MD4 hashes.  Hex file input.");
    strcpy(this->HashTypes[CH_HASH_TYPE_MD4_PLAIN].HashAlgorithm, "md4($pass)");
    this->HashTypes[CH_HASH_TYPE_MD4_PLAIN].DefaultWorkunitSize = 32;
    this->HashTypes[CH_HASH_TYPE_MD4_PLAIN].MinSupportedLength = 1;
    this->HashTypes[CH_HASH_TYPE_MD4_PLAIN].MaxSupportedLength = 16;
    this->HashTypes[CH_HASH_TYPE_MD4_PLAIN].NetworkSupportEnabled = 1;
    this->HashTypes[CH_HASH_TYPE_MD4_PLAIN].MaxHashCount = 0;

    // Hash type NTLM
    // CHHashTypePlainNTLM
    strcpy(this->HashTypes[CH_HASH_TYPE_NTLM].HashString, "NTLM");
    strcpy(this->HashTypes[CH_HASH_TYPE_NTLM].HashDescription, "Plain unsalted NTLM hashes.  Standard Windows type.  Hex file input.");
    strcpy(this->HashTypes[CH_HASH_TYPE_NTLM].HashAlgorithm, "md4(utf16-le($pass))");
    this->HashTypes[CH_HASH_TYPE_NTLM].DefaultWorkunitSize = 32;
    this->HashTypes[CH_HASH_TYPE_NTLM].MinSupportedLength = 1;
    this->HashTypes[CH_HASH_TYPE_NTLM].MaxSupportedLength = 27;
    this->HashTypes[CH_HASH_TYPE_NTLM].NetworkSupportEnabled = 1;
    this->HashTypes[CH_HASH_TYPE_NTLM].MaxHashCount = 0;

    // Hash type SHA1
    // CHHashTypePlainSHA1
    strcpy(this->HashTypes[CH_HASH_TYPE_SHA1_PLAIN].HashString, "SHA1");
    strcpy(this->HashTypes[CH_HASH_TYPE_SHA1_PLAIN].HashDescription, "Plain unsalted SHA1 hashes.  Hex file input.");
    strcpy(this->HashTypes[CH_HASH_TYPE_SHA1_PLAIN].HashAlgorithm, "sha1($pass)");
    this->HashTypes[CH_HASH_TYPE_SHA1_PLAIN].DefaultWorkunitSize = 32;
    this->HashTypes[CH_HASH_TYPE_SHA1_PLAIN].MinSupportedLength = 1;
    this->HashTypes[CH_HASH_TYPE_SHA1_PLAIN].MaxSupportedLength = 48;
    this->HashTypes[CH_HASH_TYPE_SHA1_PLAIN].NetworkSupportEnabled = 1;
    this->HashTypes[CH_HASH_TYPE_SHA1_PLAIN].MaxHashCount = 0;

    // Hash type MSSQL
    // CHHashTypeMSSQL
    strcpy(this->HashTypes[CH_HASH_TYPE_MSSQL].HashString, "MSSQL");
    strcpy(this->HashTypes[CH_HASH_TYPE_MSSQL].HashDescription, "MSSQL hashes.  Standard hash output format (0100 header), one per line.");
    strcpy(this->HashTypes[CH_HASH_TYPE_MSSQL].HashAlgorithm, "sha1($salt.strtoupper($pass))");
    this->HashTypes[CH_HASH_TYPE_MSSQL].DefaultWorkunitSize = 30;
    this->HashTypes[CH_HASH_TYPE_MSSQL].MinSupportedLength = 1;
    this->HashTypes[CH_HASH_TYPE_MSSQL].MaxSupportedLength = 16;
    this->HashTypes[CH_HASH_TYPE_MSSQL].NetworkSupportEnabled = 0;
    this->HashTypes[CH_HASH_TYPE_MSSQL].MaxHashCount = MAX_MSSQL_HASHES;

    /* TODO: FIX MYSQL323 SUPPORT
    // Hash type MySQL323
    // CHHashTypeMySQL323
    strcpy(this->HashTypes[CH_HASH_TYPE_MSSQL].HashString, "MYSQL323");
    strcpy(this->HashTypes[CH_HASH_TYPE_MSSQL].HashDescription, "MySQL old hashes.");
    strcpy(this->HashTypes[CH_HASH_TYPE_MSSQL].HashAlgorithm, "mysql323($pass)");
    this->HashTypes[CH_HASH_TYPE_MSSQL].DefaultWorkunitSize = 36;
    this->HashTypes[CH_HASH_TYPE_MSSQL].MinSupportedLength = 1;
    this->HashTypes[CH_HASH_TYPE_MSSQL].MaxSupportedLength = 16;
    this->HashTypes[CH_HASH_TYPE_MSSQL].NetworkSupportEnabled = 0;
    this->HashTypes[CH_HASH_TYPE_MSSQL].MaxHashCount = MAX_MSSQL_HASHES;
    */

    // Hash type {SHA}
    // CHHashTypePlainSHA1
    strcpy(this->HashTypes[CH_HASH_TYPE_SHA].HashString, "SHA");
    strcpy(this->HashTypes[CH_HASH_TYPE_SHA].HashDescription, "Base64 encoded {SHA} hashes");
    strcpy(this->HashTypes[CH_HASH_TYPE_SHA].HashAlgorithm, "sha1(base64decode($pass))");
    this->HashTypes[CH_HASH_TYPE_SHA].DefaultWorkunitSize = 32;
    this->HashTypes[CH_HASH_TYPE_SHA].MinSupportedLength = 1;
    this->HashTypes[CH_HASH_TYPE_SHA].MaxSupportedLength = 48;
    this->HashTypes[CH_HASH_TYPE_SHA].NetworkSupportEnabled = 0;
    this->HashTypes[CH_HASH_TYPE_SHA].MaxHashCount = 0;

    // Hash type MD5_PASS_SALT
    // CHHashTypeSaltedMD5PassSalt
    strcpy(this->HashTypes[CH_HASH_TYPE_MD5_PASS_SALT].HashString, "MD5_PS");
    strcpy(this->HashTypes[CH_HASH_TYPE_MD5_PASS_SALT].HashDescription, "Salted MD5 passwords of the pass.salt variety");
    strcpy(this->HashTypes[CH_HASH_TYPE_MD5_PASS_SALT].HashAlgorithm, "md5($pass.$salt)");
    this->HashTypes[CH_HASH_TYPE_MD5_PASS_SALT].DefaultWorkunitSize = 32;
    this->HashTypes[CH_HASH_TYPE_MD5_PASS_SALT].MinSupportedLength = 1;
    this->HashTypes[CH_HASH_TYPE_MD5_PASS_SALT].MaxSupportedLength = 16;
    this->HashTypes[CH_HASH_TYPE_MD5_PASS_SALT].NetworkSupportEnabled = 0;
    this->HashTypes[CH_HASH_TYPE_MD5_PASS_SALT].MaxHashCount = MAX_SALTED_HASHES;

    // Hash type MD5_SALT_PASS
    // CHHashTypeSaltedMD5SaltPass
    strcpy(this->HashTypes[CH_HASH_TYPE_MD5_SALT_PASS].HashString, "MD5_SP");
    strcpy(this->HashTypes[CH_HASH_TYPE_MD5_SALT_PASS].HashDescription, "Salted MD5 passwords of the salt.pass variety");
    strcpy(this->HashTypes[CH_HASH_TYPE_MD5_SALT_PASS].HashAlgorithm, "md5($salt.$pass)");
    this->HashTypes[CH_HASH_TYPE_MD5_SALT_PASS].DefaultWorkunitSize = 32;
    this->HashTypes[CH_HASH_TYPE_MD5_SALT_PASS].MinSupportedLength = 1;
    this->HashTypes[CH_HASH_TYPE_MD5_SALT_PASS].MaxSupportedLength = 16;
    this->HashTypes[CH_HASH_TYPE_MD5_SALT_PASS].NetworkSupportEnabled = 0;
    this->HashTypes[CH_HASH_TYPE_MD5_SALT_PASS].MaxHashCount = MAX_SALTED_HASHES;

    // Hash type {SSHA}
    // CHHashTypeSaltedSSHA
    strcpy(this->HashTypes[CH_HASH_TYPE_SSHA].HashString, "SSHA");
    strcpy(this->HashTypes[CH_HASH_TYPE_SSHA].HashDescription, "Base64 encoded {SSHA} salted hashes");
    strcpy(this->HashTypes[CH_HASH_TYPE_SSHA].HashAlgorithm, "sha1(base64decode($pass).$salt)");
    this->HashTypes[CH_HASH_TYPE_SSHA].DefaultWorkunitSize = 32;
    this->HashTypes[CH_HASH_TYPE_SSHA].MinSupportedLength = 1;
    this->HashTypes[CH_HASH_TYPE_SSHA].MaxSupportedLength = 16;
    this->HashTypes[CH_HASH_TYPE_SSHA].NetworkSupportEnabled = 0;
    this->HashTypes[CH_HASH_TYPE_SSHA].MaxHashCount = MAX_SALTED_HASHES;

    strcpy(this->HashTypes[CH_HASH_TYPE_MD5_SINGLE].HashString, "MD5S");
    strcpy(this->HashTypes[CH_HASH_TYPE_MD5_SINGLE].HashDescription, "Single unsalted MD5 hash.");
    strcpy(this->HashTypes[CH_HASH_TYPE_MD5_SINGLE].HashAlgorithm, "md5($pass)");
    this->HashTypes[CH_HASH_TYPE_MD5_SINGLE].DefaultWorkunitSize = 32;
    this->HashTypes[CH_HASH_TYPE_MD5_SINGLE].MinSupportedLength = 1;
    this->HashTypes[CH_HASH_TYPE_MD5_SINGLE].MaxSupportedLength = 16;
    this->HashTypes[CH_HASH_TYPE_MD5_SINGLE].NetworkSupportEnabled = 1;
    this->HashTypes[CH_HASH_TYPE_MD5_SINGLE].MaxHashCount = 1;
    
    strcpy(this->HashTypes[CH_HASH_TYPE_DOUBLE_MD5].HashString, "DOUBLEMD5");
    strcpy(this->HashTypes[CH_HASH_TYPE_DOUBLE_MD5].HashDescription, "Unsalted double MD5 hash.");
    strcpy(this->HashTypes[CH_HASH_TYPE_DOUBLE_MD5].HashAlgorithm, "md5(md5($pass))");
    this->HashTypes[CH_HASH_TYPE_DOUBLE_MD5].DefaultWorkunitSize = 32;
    this->HashTypes[CH_HASH_TYPE_DOUBLE_MD5].MinSupportedLength = 1;
    this->HashTypes[CH_HASH_TYPE_DOUBLE_MD5].MaxSupportedLength = 48;
    this->HashTypes[CH_HASH_TYPE_DOUBLE_MD5].NetworkSupportEnabled = 1;
    this->HashTypes[CH_HASH_TYPE_DOUBLE_MD5].MaxHashCount = 0;

    strcpy(this->HashTypes[CH_HASH_TYPE_MD5_OF_SHA1].HashString, "MD5OFSHA1");
    strcpy(this->HashTypes[CH_HASH_TYPE_MD5_OF_SHA1].HashDescription, "MD5 of SHA1");
    strcpy(this->HashTypes[CH_HASH_TYPE_MD5_OF_SHA1].HashAlgorithm, "md5(sha1($pass))");
    this->HashTypes[CH_HASH_TYPE_MD5_OF_SHA1].DefaultWorkunitSize = 32;
    this->HashTypes[CH_HASH_TYPE_MD5_OF_SHA1].MinSupportedLength = 1;
    this->HashTypes[CH_HASH_TYPE_MD5_OF_SHA1].MaxSupportedLength = 16;
    this->HashTypes[CH_HASH_TYPE_MD5_OF_SHA1].NetworkSupportEnabled = 1;
    this->HashTypes[CH_HASH_TYPE_MD5_OF_SHA1].MaxHashCount = 0;

    strcpy(this->HashTypes[CH_HASH_TYPE_SHA1_OF_MD5].HashString, "SHA1OFMD5");
    strcpy(this->HashTypes[CH_HASH_TYPE_SHA1_OF_MD5].HashDescription, "SHA1 of MD5");
    strcpy(this->HashTypes[CH_HASH_TYPE_SHA1_OF_MD5].HashAlgorithm, "sha1(md5($pass))");
    this->HashTypes[CH_HASH_TYPE_SHA1_OF_MD5].DefaultWorkunitSize = 32;
    this->HashTypes[CH_HASH_TYPE_SHA1_OF_MD5].MinSupportedLength = 1;
    this->HashTypes[CH_HASH_TYPE_SHA1_OF_MD5].MaxSupportedLength = 16;
    this->HashTypes[CH_HASH_TYPE_SHA1_OF_MD5].NetworkSupportEnabled = 1;
    this->HashTypes[CH_HASH_TYPE_SHA1_OF_MD5].MaxHashCount = 0;

    strcpy(this->HashTypes[CH_HASH_TYPE_TRIPLE_MD5].HashString, "TRIPLEMD5");
    strcpy(this->HashTypes[CH_HASH_TYPE_TRIPLE_MD5].HashDescription, "Unsalted triple MD5 hash.");
    strcpy(this->HashTypes[CH_HASH_TYPE_TRIPLE_MD5].HashAlgorithm, "md5(md5(md5($pass)))");
    this->HashTypes[CH_HASH_TYPE_TRIPLE_MD5].DefaultWorkunitSize = 32;
    this->HashTypes[CH_HASH_TYPE_TRIPLE_MD5].MinSupportedLength = 1;
    this->HashTypes[CH_HASH_TYPE_TRIPLE_MD5].MaxSupportedLength = 48;
    this->HashTypes[CH_HASH_TYPE_TRIPLE_MD5].NetworkSupportEnabled = 1;
    this->HashTypes[CH_HASH_TYPE_TRIPLE_MD5].MaxHashCount = 0;

    strcpy(this->HashTypes[CH_HASH_TYPE_DUPLICATED_MD5].HashString, "DUPMD5");
    strcpy(this->HashTypes[CH_HASH_TYPE_DUPLICATED_MD5].HashDescription, "MD5 of a doubled password");
    strcpy(this->HashTypes[CH_HASH_TYPE_DUPLICATED_MD5].HashAlgorithm, "md5($pass.$pass)");
    this->HashTypes[CH_HASH_TYPE_DUPLICATED_MD5].DefaultWorkunitSize = 32;
    this->HashTypes[CH_HASH_TYPE_DUPLICATED_MD5].MinSupportedLength = 1;
    this->HashTypes[CH_HASH_TYPE_DUPLICATED_MD5].MaxSupportedLength = 24;
    this->HashTypes[CH_HASH_TYPE_DUPLICATED_MD5].NetworkSupportEnabled = 1;
    this->HashTypes[CH_HASH_TYPE_DUPLICATED_MD5].MaxHashCount = 0;

    strcpy(this->HashTypes[CH_HASH_TYPE_DUPLICATED_NTLM].HashString, "DUPNTLM");
    strcpy(this->HashTypes[CH_HASH_TYPE_DUPLICATED_NTLM].HashDescription, "NTLM of a doubled password");
    strcpy(this->HashTypes[CH_HASH_TYPE_DUPLICATED_NTLM].HashAlgorithm, "ntlm(utf16le($pass.$pass))");
    this->HashTypes[CH_HASH_TYPE_DUPLICATED_NTLM].DefaultWorkunitSize = 32;
    this->HashTypes[CH_HASH_TYPE_DUPLICATED_NTLM].MinSupportedLength = 1;
    this->HashTypes[CH_HASH_TYPE_DUPLICATED_NTLM].MaxSupportedLength = 13;
    this->HashTypes[CH_HASH_TYPE_DUPLICATED_NTLM].NetworkSupportEnabled = 1;
    this->HashTypes[CH_HASH_TYPE_DUPLICATED_NTLM].MaxHashCount = 0;

    strcpy(this->HashTypes[CH_HASH_TYPE_LM].HashString, "LM");
    strcpy(this->HashTypes[CH_HASH_TYPE_LM].HashDescription, "LM hash");
    strcpy(this->HashTypes[CH_HASH_TYPE_LM].HashAlgorithm, "LM($pass)");
    this->HashTypes[CH_HASH_TYPE_LM].DefaultWorkunitSize = 32;
    this->HashTypes[CH_HASH_TYPE_LM].MinSupportedLength = 1;
    this->HashTypes[CH_HASH_TYPE_LM].MaxSupportedLength = 7;
    this->HashTypes[CH_HASH_TYPE_LM].NetworkSupportEnabled = 1;
    this->HashTypes[CH_HASH_TYPE_LM].MaxHashCount = 0;

    strcpy(this->HashTypes[CH_HASH_TYPE_SHA256].HashString, "SHA256");
    strcpy(this->HashTypes[CH_HASH_TYPE_SHA256].HashDescription, "SHA256 hash");
    strcpy(this->HashTypes[CH_HASH_TYPE_SHA256].HashAlgorithm, "sha256($pass)");
    this->HashTypes[CH_HASH_TYPE_SHA256].DefaultWorkunitSize = 32;
    this->HashTypes[CH_HASH_TYPE_SHA256].MinSupportedLength = 0;
    this->HashTypes[CH_HASH_TYPE_SHA256].MaxSupportedLength = 48;
    this->HashTypes[CH_HASH_TYPE_SHA256].NetworkSupportEnabled = 1;
    this->HashTypes[CH_HASH_TYPE_SHA256].MaxHashCount = 0;

    this->NumberOfHashes = MAX_HASH_ID_VALUE;
}

int CHHashes::GetHashIdFromString(const char* HashString) {
    int i;
    fflush(stdout);
    for (i = 0; i < this->NumberOfHashes; i++) {
        if (strcmp(HashString, this->HashTypes[i].HashString) == 0) {
            this->CurrentHashId = i;
            return i;
        }
    }
    return -1;
}

int CHHashes::GetNumberOfHashes() {
    return this->NumberOfHashes;
}

char *CHHashes::GetHashStringFromID(int hashId) {
    return this->HashTypes[hashId].HashString;
}
int CHHashes::GetHashId() {
    return this->CurrentHashId;
}

// Returns 0 if hash is not set.
uint8_t CHHashes::GetMinSupportedLength() {
    if (this->CurrentHashId == -1) {
        return 0;
    } else {
        return this->HashTypes[this->CurrentHashId].MinSupportedLength;
    }
}

uint8_t CHHashes::GetMaxSupportedLength() {
   if (this->CurrentHashId == -1) {
        return 0;
    } else {
        return this->HashTypes[this->CurrentHashId].MaxSupportedLength;
    }
}

uint8_t CHHashes::GetIsNetworkSupported() {
    if (this->CurrentHashId == -1) {
        return 0;
    } else {
        return this->HashTypes[this->CurrentHashId].NetworkSupportEnabled;
    }
}

uint8_t CHHashes::GetDefaultWorkunitSizeBits() {
    if (this->CurrentHashId == -1) {
        return 32;
    } else {
        return this->HashTypes[this->CurrentHashId].DefaultWorkunitSize;
    }
}

uint32_t CHHashes::GetMaxHashCount() {
    if (this->CurrentHashId == -1) {
        return 0;
    } else {
        return this->HashTypes[this->CurrentHashId].MaxHashCount;
    }
}

CHHashFileTypes *CHHashes::GetHashFile() {

    CHHashFileTypes *HashFile = NULL;

    // Cleaned this up - went to fallthrough on the case statement.
    switch(this->CurrentHashId) {

        case CH_HASH_TYPE_MD5_PLAIN: // MD5
        case CH_HASH_TYPE_MD4_PLAIN: // MD4
        case CH_HASH_TYPE_NTLM: // NTLM
        case CH_HASH_TYPE_MD5_SINGLE: // MD5 Single
        case CH_HASH_TYPE_DOUBLE_MD5:
        case CH_HASH_TYPE_MD5_OF_SHA1:
        case CH_HASH_TYPE_TRIPLE_MD5:
        case CH_HASH_TYPE_DUPLICATED_MD5:
        case CH_HASH_TYPE_DUPLICATED_NTLM:
            HashFile = new CHHashFilePlain32(16);
            break;
        case CH_HASH_TYPE_SHA1_PLAIN: // SHA1 (unsalted)
        case CH_HASH_TYPE_SHA1_OF_MD5:
            HashFile = new CHHashFilePlain32(20);
            break;
        case CH_HASH_TYPE_MSSQL: // MSSQL
            HashFile = new CHHashFileMSSQL();
            break;
        case CH_HASH_TYPE_MYSQL323: // MySQL 323 (old mysql)
            HashFile = new CHHashFilePlain32(8);
            break;
        case CH_HASH_TYPE_SHA: //{SHA}
            HashFile = new CHHashFilePlainSHA();
            break;
        case CH_HASH_TYPE_MD5_PASS_SALT: //MD5 Salted PassSalt
            HashFile = new CHHashFileSalted32(MD5_HASH_LENGTH, MAX_SALT_LENGTH,
                SALT_IS_LAST, SALT_IS_LITERAL);
            break;
        case CH_HASH_TYPE_MD5_SALT_PASS: //MD5 Salted SaltPass
            HashFile = new CHHashFileSalted32(MD5_HASH_LENGTH, MAX_SALT_LENGTH,
                SALT_IS_LAST, SALT_IS_LITERAL);
            break;
        case CH_HASH_TYPE_SSHA: //{SSHA}
            HashFile = new CHHashFileSaltedSSHA();
            break;
        case CH_HASH_TYPE_LM: // Will need new hash file type for this!
            HashFile = new CHHashFileLM();
            //HashFile = new CHHashFilePlain32(8);
            break;
        case CH_HASH_TYPE_SHA256:
            HashFile = new CHHashFilePlain32(32);
            break;
        default:
            printf("Hash type not supported yet!\n");
            exit(1);
    }
    return HashFile;
}

CHHashType *CHHashes::GetHashType() {

    CHHashType *HashType = NULL;


    switch(this->CurrentHashId) {
    case CH_HASH_TYPE_MD5_PLAIN: // MD5
        HashType = new CHHashTypePlainMD5();
        break;
    case CH_HASH_TYPE_MD4_PLAIN: // MD4
        HashType = new CHHashTypePlainMD4();
        break;
    case CH_HASH_TYPE_NTLM: // NTLM
        HashType = new CHHashTypePlainNTLM();
        break;
    case CH_HASH_TYPE_SHA1_PLAIN: // SHA1 (unsalted)
        HashType = new CHHashTypePlainSHA1();
        break;
    case CH_HASH_TYPE_MSSQL: // MSSQL
        HashType = new CHHashTypeMSSQL();
        break;
    case CH_HASH_TYPE_MYSQL323: // MySQL 323 (old mysql)
        HashType = new CHHashTypePlainMySQL323();
        break;
    case CH_HASH_TYPE_SHA: //{SHA}
        HashType = new CHHashTypePlainSHA1();
        break;
    case CH_HASH_TYPE_MD5_PASS_SALT: //MD5 Salted PassSalt
        HashType = new CHHashTypeSaltedMD5PassSalt();
        break;
    case CH_HASH_TYPE_MD5_SALT_PASS: //MD5 Salted SaltPass
        HashType = new CHHashTypeSaltedMD5SaltPass();
        break;
    case CH_HASH_TYPE_SSHA: //{SSHA}
        HashType = new CHHashTypeSaltedSSHA();
        break;
    case CH_HASH_TYPE_MD5_SINGLE: // MD5 Single
        HashType = new CHHashTypePlainMD5Single();
        break;
    case CH_HASH_TYPE_DOUBLE_MD5:
        HashType = new CHHashTypePlainDoubleMD5();
        break;
    case CH_HASH_TYPE_MD5_OF_SHA1:
        HashType = new CHHashTypePlainMD5OfSHA1();
        break;
    case CH_HASH_TYPE_SHA1_OF_MD5:
        HashType = new CHHashTypePlainSHA1OfMD5();
        break;
    case CH_HASH_TYPE_TRIPLE_MD5:
        HashType = new CHHashTypePlainTripleMD5();
        break;
    case CH_HASH_TYPE_DUPLICATED_MD5:
        HashType = new CHHashTypePlainDuplicatedMD5();
        break;
    case CH_HASH_TYPE_DUPLICATED_NTLM:
        HashType = new CHHashTypePlainDuplicatedNTLM();
        break;
    case CH_HASH_TYPE_LM:
        HashType = new CHHashTypePlainLM();
        break;
    case CH_HASH_TYPE_SHA256:
        HashType = new CHHashTypePlainSHA256();
        break;
    default:
        printf("Hash type not supported yet!\n");
        exit(1);
        }

    return HashType;
}

void CHHashes::SetHashId(int newHashId) {
    this->CurrentHashId = newHashId;
}

void CHHashes::PrintAllHashTypes() {
    // Print all current hashes in the system.

    int i;

    int supportedHashCount = 0;

    printf("\n\n");
    for (i = 0; i < MAX_HASH_ID_VALUE; i++) {
        // If the hash string is non-null, print the stuff out.
        if (this->HashTypes[i].HashString[0]) {
            supportedHashCount++;
            printf("Hash type:       %s\n", this->HashTypes[i].HashString);
            printf("Algorithm:       %s\n", this->HashTypes[i].HashAlgorithm);
            printf("Description:     %s\n", this->HashTypes[i].HashDescription);
            printf("Min length:      %d\n", this->HashTypes[i].MinSupportedLength);
            printf("Max length:      %d\n", this->HashTypes[i].MaxSupportedLength);
            printf("Network support: %s\n", this->HashTypes[i].NetworkSupportEnabled ? "Yes" : "No");
            if (this->HashTypes[i].MaxHashCount) {
                printf("Max hash count:  %d\n", this->HashTypes[i].MaxHashCount);
            } else {
                printf("Max hash count:  Unlimited\n");
            }
            printf("\n\n");
        }
    }
    printf("Currently supported hash types: %d\n\n", supportedHashCount);
}

/*
 *     strcpy(this->HashTypes[CH_HASH_TYPE_SHA256].HashString, "SHA256");
    strcpy(this->HashTypes[CH_HASH_TYPE_SHA256].HashDescription, "SHA256 hash");
    strcpy(this->HashTypes[CH_HASH_TYPE_SHA256].HashAlgorithm, "sha256($pass)");
    this->HashTypes[CH_HASH_TYPE_SHA256].DefaultWorkunitSize = 32;
    this->HashTypes[CH_HASH_TYPE_SHA256].MinSupportedLength = 0;
    this->HashTypes[CH_HASH_TYPE_SHA256].MaxSupportedLength = 48;
    this->HashTypes[CH_HASH_TYPE_SHA256].NetworkSupportEnabled = 1;
    this->HashTypes[CH_HASH_TYPE_SHA256].MaxHashCount = 0;
*/
