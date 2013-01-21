// Test diamond derived classes

#include <stdio.h>
#include <stdlib.h>

class HashType {
public:
    HashType(int hashTypeLength) {
        printf("HashType::HashType(%d)\n", hashTypeLength);
    }
};

class HashTypePlain : public HashType {
public:
    HashTypePlain(int hashTypePlainLength);
};

HashTypePlain::HashTypePlain(int hashTypePlainLength) : HashType(hashTypePlainLength) {
    printf("HashTypePlain::HashTypePlain(%d)\n", hashTypePlainLength);
}

class HashTypePlainCUDA : public virtual HashTypePlain {
public:
    HashTypePlainCUDA(int hashTypePlainCUDALength);
};
HashTypePlainCUDA::HashTypePlainCUDA(int hashTypePlainCUDALength) : HashTypePlain(hashTypePlainCUDALength) {
    printf("HashTypePlainCUDA::HashTypePlainCUDA(%d)\n", hashTypePlainCUDALength);
}

class HashTypeSaltedCUDA : public HashTypePlainCUDA {
public:
    HashTypeSaltedCUDA(int hashTypeSaltedCUDALength);
};
HashTypeSaltedCUDA::HashTypeSaltedCUDA(int hashTypeSaltedCUDALength) : HashTypePlainCUDA(hashTypeSaltedCUDALength), HashTypePlain(hashTypeSaltedCUDALength) {
    printf("HashTypeSaltedCUDA::HashTypeSaltedCUDA(%d)\n", hashTypeSaltedCUDALength);
}

class HashTypeSalted : public virtual HashTypePlain {
public:
    HashTypeSalted(int hashTypeSaltedLength);
};
HashTypeSalted::HashTypeSalted(int hashTypeSaltedLength) : HashTypePlain(hashTypeSaltedLength) {
    printf("HashTypeSalted::HashTypeSalted(%d)\n", hashTypeSaltedLength);
}

class HashTypePlainCudaMD5 : public HashTypePlainCUDA {
public:
    HashTypePlainCudaMD5();
};

HashTypePlainCudaMD5::HashTypePlainCudaMD5() : HashTypePlainCUDA(16), HashTypePlain(16) {
    printf("HashTypePlainCudaMD5::HashTypePlainCudaMD5()\n");
}

class HashTypeSaltedCudaMD5 : public HashTypeSaltedCUDA, public HashTypeSalted {
public:
    HashTypeSaltedCudaMD5();
};

HashTypeSaltedCudaMD5::HashTypeSaltedCudaMD5() : HashTypeSaltedCUDA(16), HashTypeSalted(16), HashTypePlain(16) {
    printf("HashTypeSaltedCudaMD5::HashTypeSaltedCudaMD5()\n");
}


int main() {
    HashType *Foo;
    
    Foo = new HashTypePlainCudaMD5();
    delete Foo;
    printf("\n\n");
    Foo = new HashTypeSaltedCudaMD5();
    
}