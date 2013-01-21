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

#include "MFN_Common/MFNHashIdentifiers.h"
#include <stdio.h>


MFNHashIdentifiers::MFNHashIdentifiers() {
    this->CurrentHashId = MFN_HASHTYPE_UNDEFINED;
    this->CurrentHashPosition = 0;
    
    MFNHashIdentifierData NewHash;
    uint32_t i;
    
    // Null identifier - this is for values returned if nothing is set.
    // Doing this allows us to leave CurrentHashPosition at 0 and return
    // sane numbers.

    NewHash.HashDescriptor = "UNDEFINED";
    NewHash.HashID = MFN_HASHTYPE_UNDEFINED;
    NewHash.HashDetails = "Placeholder hash type.";
    NewHash.HashAlgorithm = "NULL";
    NewHash.DefaultWorkunitSize = 32;
    NewHash.MinSupportedLength = 0;
    NewHash.MaxSupportedLength = 0;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 0;
    NewHash.HasCPUSupport = 0;
    NewHash.HasCUDASupport = 0;
    NewHash.HasOpenCLSupport = 0;
    NewHash.HashTypeIdentifier = 0;
    NewHash.HashFileIdentifier = 0;
    NewHash.HasWordlistSupport = 0;
    this->SupportedHashTypes.push_back(NewHash);
    
    // MD5, MFN_HASHTYPE_PLAIN_MD5
    NewHash.HashDescriptor = "MD5";
    NewHash.HashID = MFN_HASHTYPE_PLAIN_MD5;
    NewHash.HashDetails = "Plain unsalted MD5 hashes.";
    NewHash.HashAlgorithm = "md5($pass)";
    NewHash.DefaultWorkunitSize = 34;
    NewHash.MinSupportedLength = 1;
    NewHash.MaxSupportedLength = 55;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 0;
    NewHash.HasCPUSupport = 1;
    NewHash.HasCUDASupport = 1;
    NewHash.HasOpenCLSupport = 1;
    NewHash.HashTypeIdentifier = MFN_HASHTYPE_PLAIN_MD5;
    NewHash.HashFileIdentifier = CH_HASHFILE_PLAIN_16;
    NewHash.HasWordlistSupport = 0;
    this->SupportedHashTypes.push_back(NewHash);

    // MD5, MFN_HASHTYPE_PLAIN_MD5
    NewHash.HashDescriptor = "SMD5";
    NewHash.HashID = MFN_HASHTYPE_PLAIN_MD5_SINGLE;
    NewHash.HashDetails = "Plain unsalted MD5 hashes.";
    NewHash.HashAlgorithm = "md5($pass)";
    NewHash.DefaultWorkunitSize = 36;
    NewHash.MinSupportedLength = 1;
    NewHash.MaxSupportedLength = 15;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 0;
    NewHash.HasCPUSupport = 1;
    NewHash.HasCUDASupport = 1;
    NewHash.HasOpenCLSupport = 1;
    NewHash.HashTypeIdentifier = MFN_HASHTYPE_PLAIN_MD5_SINGLE;
    NewHash.HashFileIdentifier = CH_HASHFILE_PLAIN_16;
    NewHash.HasWordlistSupport = 0;
    this->SupportedHashTypes.push_back(NewHash);

    // NTLM, MFN_HASHTYPE_NTLM
    NewHash.HashDescriptor = "NTLM";
    NewHash.HashID = MFN_HASHTYPE_NTLM;
    NewHash.HashDetails = "NTLM hashes.";
    NewHash.HashAlgorithm = "md4(utf16-le($pass))";
    NewHash.DefaultWorkunitSize = 34;
    NewHash.MinSupportedLength = 1;
    NewHash.MaxSupportedLength = 27;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 0;
    NewHash.HasCPUSupport = 1;
    NewHash.HasCUDASupport = 1;
    NewHash.HasOpenCLSupport = 1;
    NewHash.HashTypeIdentifier = MFN_HASHTYPE_NTLM;
    NewHash.HashFileIdentifier = CH_HASHFILE_PLAIN_16;
    NewHash.HasWordlistSupport = 0;
    this->SupportedHashTypes.push_back(NewHash);

    // NTLM, MFN_HASHTYPE_NTLM_SINGLE
    NewHash.HashDescriptor = "SNTLM";
    NewHash.HashID = MFN_HASHTYPE_NTLM_SINGLE;
    NewHash.HashDetails = "NTLM hashes.";
    NewHash.HashAlgorithm = "md4(utf16-le($pass))";
    NewHash.DefaultWorkunitSize = 36;
    NewHash.MinSupportedLength = 1;
    NewHash.MaxSupportedLength = 15;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 1;
    NewHash.HasCPUSupport = 1;
    NewHash.HasCUDASupport = 1;
    NewHash.HasOpenCLSupport = 1;
    NewHash.HashTypeIdentifier = MFN_HASHTYPE_NTLM_SINGLE;
    NewHash.HashFileIdentifier = CH_HASHFILE_PLAIN_16;
    NewHash.HasWordlistSupport = 0;
    this->SupportedHashTypes.push_back(NewHash);

    // LM, MFN_HASHTYPE_LM
    NewHash.HashDescriptor = "LM";
    NewHash.HashID = MFN_HASHTYPE_LM;
    NewHash.HashDetails = "LM hashes.";
    NewHash.HashAlgorithm = "LanMan Algorithm";
    NewHash.DefaultWorkunitSize = 32;
    NewHash.MinSupportedLength = 1;
    NewHash.MaxSupportedLength = 7;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 0;
    NewHash.HasCPUSupport = 0;
    NewHash.HasCUDASupport = 1;
    NewHash.HasOpenCLSupport = 0;
    NewHash.HashTypeIdentifier = MFN_HASHTYPE_LM;
    NewHash.HashFileIdentifier = CH_HASHFILE_LM;
    NewHash.HasWordlistSupport = 0;
    this->SupportedHashTypes.push_back(NewHash);

    // DOUBLEMD5, MFN_HASHTYPE_PLAIN_MD5
    NewHash.HashDescriptor = "DOUBLEMD5";
    NewHash.HashID = MFN_HASHTYPE_DOUBLE_MD5;
    NewHash.HashDetails = "Double unsalted MD5 hashes.";
    NewHash.HashAlgorithm = "md5(md5($pass))";
    NewHash.DefaultWorkunitSize = 32;
    NewHash.MinSupportedLength = 1;
    NewHash.MaxSupportedLength = 8;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 0;
    NewHash.HasCPUSupport = 1;
    NewHash.HasCUDASupport = 1;
    NewHash.HasOpenCLSupport = 1;
    NewHash.HashTypeIdentifier = MFN_HASHTYPE_DOUBLE_MD5;
    NewHash.HashFileIdentifier = CH_HASHFILE_PLAIN_16;
    NewHash.HasWordlistSupport = 0;
    this->SupportedHashTypes.push_back(NewHash);

    // 20 byte SHA1 hash
    NewHash.HashDescriptor = "SHA1";
    NewHash.HashID = MFN_HASHTYPE_SHA1;
    NewHash.HashDetails = "Unsalted SHA1 hashes.";
    NewHash.HashAlgorithm = "sha1($pass)";
    NewHash.DefaultWorkunitSize = 32;
    NewHash.MinSupportedLength = 1;
    NewHash.MaxSupportedLength = 8;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 0;
    NewHash.HasCPUSupport = 1;
    NewHash.HasCUDASupport = 1;
    NewHash.HasOpenCLSupport = 1;
    NewHash.HashTypeIdentifier = MFN_HASHTYPE_SHA1;
    NewHash.HashFileIdentifier = CH_HASHFILE_PLAIN_20;
    NewHash.HasWordlistSupport = 0;
    this->SupportedHashTypes.push_back(NewHash);

    // 32-byte SHA256 hash
    NewHash.HashDescriptor = "SHA256";
    NewHash.HashID = MFN_HASHTYPE_SHA256;
    NewHash.HashDetails = "Unsalted SHA256 hashes.";
    NewHash.HashAlgorithm = "sha256($pass)";
    NewHash.DefaultWorkunitSize = 32;
    NewHash.MinSupportedLength = 1;
    NewHash.MaxSupportedLength = 8;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 0;
    NewHash.HasCPUSupport = 1;
    NewHash.HasCUDASupport = 1;
    NewHash.HasOpenCLSupport = 1;
    NewHash.HashTypeIdentifier = MFN_HASHTYPE_SHA256;
    NewHash.HashFileIdentifier = CH_HASHFILE_PLAIN_32;
    NewHash.HasWordlistSupport = 0;
    this->SupportedHashTypes.push_back(NewHash);

    // 32-byte Double SHA256 hash
    NewHash.HashDescriptor = "DOUBLESHA256";
    NewHash.HashID = MFN_HASHTYPE_DOUBLE_SHA256;
    NewHash.HashDetails = "Unsalted Double SHA256 hashes - ASCII inner hash.";
    NewHash.HashAlgorithm = "sha256(sha256($pass))";
    NewHash.DefaultWorkunitSize = 32;
    NewHash.MinSupportedLength = 1;
    NewHash.MaxSupportedLength = 8;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 0;
    NewHash.HasCPUSupport = 1;
    NewHash.HasCUDASupport = 1;
    NewHash.HasOpenCLSupport = 1;
    NewHash.HashTypeIdentifier = MFN_HASHTYPE_DOUBLE_SHA256;
    NewHash.HashFileIdentifier = CH_HASHFILE_PLAIN_32;
    NewHash.HasWordlistSupport = 0;
    this->SupportedHashTypes.push_back(NewHash);

    // Salted MD5 md5($pass.$salt) type for testing
    NewHash.HashDescriptor = "MD5_PS";
    NewHash.HashID = MFN_HASHTYPE_MD5_PS;
    NewHash.HashDetails = "Salted MD5 hashes.";
    NewHash.HashAlgorithm = "md5($pass.$salt)";
    NewHash.DefaultWorkunitSize = 32;
    NewHash.MinSupportedLength = 1;
    NewHash.MaxSupportedLength = 8;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 0;
    NewHash.HasCPUSupport = 1;
    NewHash.HasCUDASupport = 1;
    NewHash.HasOpenCLSupport = 1;
    NewHash.HashTypeIdentifier = MFN_HASHTYPE_MD5_PS;
    NewHash.HashFileIdentifier = CH_HASHFILE_SALTED_32_PASS_SALT;
    NewHash.HasWordlistSupport = 0;
    this->SupportedHashTypes.push_back(NewHash);

    // 16HEX type - multiple hashes
    NewHash.HashDescriptor = "16HEX";
    NewHash.HashID = MFN_HASHTYPE_16HEX;
    NewHash.HashDetails = "Assorted 16 byte algorithms.";
    NewHash.HashAlgorithm = "md5,md4($pass)";
    NewHash.DefaultWorkunitSize = 34;
    NewHash.MinSupportedLength = 1;
    NewHash.MaxSupportedLength = 10;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 0;
    NewHash.HasCPUSupport = 1;
    NewHash.HasCUDASupport = 1;
    NewHash.HasOpenCLSupport = 1;
    NewHash.HashTypeIdentifier = MFN_HASHTYPE_16HEX;
    NewHash.HashFileIdentifier = CH_HASHFILE_PLAIN_16;
    NewHash.HasWordlistSupport = 0;
    this->SupportedHashTypes.push_back(NewHash);

    // DUPMD5
    NewHash.HashDescriptor = "DUPMD5";
    NewHash.HashID = MFN_HASHTYPE_DUPLICATED_MD5;
    NewHash.HashDetails = "Duplicated MD5 hashes.";
    NewHash.HashAlgorithm = "md5($pass.$pass)";
    NewHash.DefaultWorkunitSize = 34;
    NewHash.MinSupportedLength = 1;
    NewHash.MaxSupportedLength = 8;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 0;
    NewHash.HasCPUSupport = 1;
    NewHash.HasCUDASupport = 1;
    NewHash.HasOpenCLSupport = 1;
    NewHash.HashTypeIdentifier = MFN_HASHTYPE_DUPLICATED_MD5;
    NewHash.HashFileIdentifier = CH_HASHFILE_PLAIN_16;
    NewHash.HasWordlistSupport = 0;
    this->SupportedHashTypes.push_back(NewHash);

    // Unsalted Lotus Notes
    NewHash.HashDescriptor = "LOTUS";
    NewHash.HashID = MFN_HASHTYPE_PLAIN_LOTUS;
    NewHash.HashDetails = "Unsalted Lotus hashes.";
    NewHash.HashAlgorithm = "md2($pass)";
    NewHash.DefaultWorkunitSize = 32;
    NewHash.MinSupportedLength = 1;
    NewHash.MaxSupportedLength = 16;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 0;
    NewHash.HasCPUSupport = 1;
    NewHash.HasCUDASupport = 1;
    NewHash.HasOpenCLSupport = 1;
    NewHash.HashTypeIdentifier = MFN_HASHTYPE_PLAIN_LOTUS;
    NewHash.HashFileIdentifier = CH_HASHFILE_PLAIN_16;
    NewHash.HasWordlistSupport = 0;
    this->SupportedHashTypes.push_back(NewHash);

    // MD5WL, MFN_HASHTYPE_PLAIN_MD5WL
    NewHash.HashDescriptor = "MD5WL";
    NewHash.HashID = MFN_HASHTYPE_PLAIN_MD5WL;
    NewHash.HashDetails = "Plain unsalted MD5 hashes, wordlist.";
    NewHash.HashAlgorithm = "md5($pass)";
    NewHash.DefaultWorkunitSize = 32;
    NewHash.MinSupportedLength = 1;
    NewHash.MaxSupportedLength = 1;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 0;
    NewHash.HasCPUSupport = 1;
    NewHash.HasCUDASupport = 1;
    NewHash.HasOpenCLSupport = 1;
    NewHash.HashTypeIdentifier = MFN_HASHTYPE_PLAIN_MD5WL;
    NewHash.HashFileIdentifier = CH_HASHFILE_PLAIN_16;
    NewHash.HasWordlistSupport = 1;
    this->SupportedHashTypes.push_back(NewHash);

    // NTLMWL, MFN_HASHTYPE_PLAIN_NTLMWL
    NewHash.HashDescriptor = "NTLMWL";
    NewHash.HashID = MFN_HASHTYPE_PLAIN_NTLMWL;
    NewHash.HashDetails = "Plain unsalted NTLM hashes, wordlist.";
    NewHash.HashAlgorithm = "md4(utf16-le($pass))";
    NewHash.DefaultWorkunitSize = 32;
    NewHash.MinSupportedLength = 1;
    NewHash.MaxSupportedLength = 1;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 0;
    NewHash.HasCPUSupport = 1;
    NewHash.HasCUDASupport = 1;
    NewHash.HasOpenCLSupport = 1;
    NewHash.HashTypeIdentifier = MFN_HASHTYPE_PLAIN_NTLMWL;
    NewHash.HashFileIdentifier = CH_HASHFILE_PLAIN_16;
    NewHash.HasWordlistSupport = 1;
    this->SupportedHashTypes.push_back(NewHash);

    // IPB md5(md5($salt).md5($pass)) type
    NewHash.HashDescriptor = "IPB";
    NewHash.HashID = MFN_HASHTYPE_IPB;
    NewHash.HashDetails = "IPB hashes.";
    NewHash.HashAlgorithm = "md5(md5($salt).md5($pass))";
    NewHash.DefaultWorkunitSize = 32;
    NewHash.MinSupportedLength = 1;
    NewHash.MaxSupportedLength = 16;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 0;
    NewHash.HasCPUSupport = 1;
    NewHash.HasCUDASupport = 1;
    NewHash.HasOpenCLSupport = 1;
    NewHash.HashTypeIdentifier = MFN_HASHTYPE_IPB;
    NewHash.HashFileIdentifier = CH_HASHFILE_IPB;
    NewHash.HasWordlistSupport = 0;
    this->SupportedHashTypes.push_back(NewHash);

    NewHash.HashDescriptor = "IPBWL";
    NewHash.HashID = MFN_HASHTYPE_IPBWL;
    NewHash.HashDetails = "IPB hashes, wordlist.";
    NewHash.HashAlgorithm = "md5(md5($salt).md5($pass))";
    NewHash.DefaultWorkunitSize = 32;
    NewHash.MinSupportedLength = 1;
    NewHash.MaxSupportedLength = 1;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 0;
    NewHash.HasCPUSupport = 1;
    NewHash.HasCUDASupport = 1;
    NewHash.HasOpenCLSupport = 1;
    NewHash.HashTypeIdentifier = MFN_HASHTYPE_IPBWL;
    NewHash.HashFileIdentifier = CH_HASHFILE_IPB;
    NewHash.HasWordlistSupport = 1;
    this->SupportedHashTypes.push_back(NewHash);

    // IPB md5(md5($salt).md5($pass)) type
    NewHash.HashDescriptor = "PHPASS";
    NewHash.HashID = MFN_HASHTYPE_PHPASS;
    NewHash.HashDetails = "Phpass hashes ('$H$' or '$P$')";
    NewHash.HashAlgorithm = "Iterated, salted, MD5";
    NewHash.DefaultWorkunitSize = 30;
    NewHash.MinSupportedLength = 1;
    NewHash.MaxSupportedLength = 16;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 0;
    NewHash.HasCPUSupport = 1;
    NewHash.HasCUDASupport = 1;
    NewHash.HasOpenCLSupport = 1;
    NewHash.HashTypeIdentifier = MFN_HASHTYPE_PHPASS;
    NewHash.HashFileIdentifier = CH_HASHFILE_PHPASS;
    NewHash.HasWordlistSupport = 0;
    this->SupportedHashTypes.push_back(NewHash);

    // IPB md5(md5($salt).md5($pass)) type
    NewHash.HashDescriptor = "PHPASSWL";
    NewHash.HashID = MFN_HASHTYPE_PHPASSWL;
    NewHash.HashDetails = "Phpass hashes ('$H$' or '$P$') with wordlist support.";
    NewHash.HashAlgorithm = "Iterated, salted, MD5";
    NewHash.DefaultWorkunitSize = 32;
    NewHash.MinSupportedLength = 1;
    NewHash.MaxSupportedLength = 16;
    NewHash.NetworkSupportEnabled = 0;
    NewHash.MaxHashCount = 0;
    NewHash.HasCPUSupport = 1;
    NewHash.HasCUDASupport = 1;
    NewHash.HasOpenCLSupport = 1;
    NewHash.HashTypeIdentifier = MFN_HASHTYPE_PHPASSWL;
    NewHash.HashFileIdentifier = CH_HASHFILE_PHPASS;
    NewHash.HasWordlistSupport = 1;
    this->SupportedHashTypes.push_back(NewHash);
}

uint32_t MFNHashIdentifiers::GetHashIdFromString(std::string HashString) {
    std::vector<MFNHashIdentifierData>::iterator hashtype;
    
    for (hashtype = this->SupportedHashTypes.begin(); 
        hashtype < this->SupportedHashTypes.end(); hashtype++) {
        // If the hash descriptor matches the string, set it and return it.
        if (hashtype->HashDescriptor == HashString) {
            this->CurrentHashId = hashtype->HashID;
            return hashtype->HashID;
        }
        // Increment the pointer to the current hash
        this->CurrentHashPosition++;
    }
    // If no hash is found, reset things and return undefined.
    this->CurrentHashPosition = 0;
    return MFN_HASHTYPE_UNDEFINED;
}

void MFNHashIdentifiers::SetHashId(uint32_t newHashId) {
    std::vector<MFNHashIdentifierData>::iterator hashtype;
    
    // Reset this to 0 each time through.
    this->CurrentHashPosition = 0;
    
    for (hashtype = this->SupportedHashTypes.begin(); 
        hashtype < this->SupportedHashTypes.end(); hashtype++) {
        // If the hash descriptor matches the string, set it and return it.
        if (hashtype->HashID == newHashId) {
            this->CurrentHashId = hashtype->HashID;
            // Break out - found the hash.
            break;
        }
        // Increment the pointer to the current hash
        this->CurrentHashPosition++;
    }
}


void MFNHashIdentifiers::PrintAllHashTypes() {

    int supportedHashCount = 0, i;

    printf("\n\n");
    printf("Supported hash types:\n\n");
    // Skip the undefined hash type
    for (i = 1; i < this->SupportedHashTypes.size(); i++) {
        supportedHashCount++;
        printf("Hash type:       %s\n", this->SupportedHashTypes[i].HashDescriptor.c_str());
        printf("Algorithm:       %s\n", this->SupportedHashTypes[i].HashAlgorithm.c_str());
        printf("Description:     %s\n", this->SupportedHashTypes[i].HashDetails.c_str());
        printf("Min length:      %d\n", this->SupportedHashTypes[i].MinSupportedLength);
        printf("Max length:      %d\n", this->SupportedHashTypes[i].MaxSupportedLength);
        printf("Network support: %s\n", this->SupportedHashTypes[i].NetworkSupportEnabled ? "Yes" : "No");
        if (this->SupportedHashTypes[i].MaxHashCount) {
            printf("Max hash count:  %d\n", this->SupportedHashTypes[i].MaxHashCount);
        } else {
            printf("Max hash count:  Unlimited\n");
        }
        printf("\n");
    }
    printf("Currently supported hash types: %d\n", supportedHashCount);
}