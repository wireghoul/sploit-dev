// Testing some static class variable stuff.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <vector>
#include <string>


class BaseClass {
public:
    BaseClass();

    virtual void runThread() = 0;

    void setStaticVar8(uint8_t newVar8) {
        this->staticVar8 = newVar8;
    }

    void setStaticVar16(uint16_t newVar16) {
        this->staticVar16 = newVar16;
    }

    void setDynamicVar8(uint8_t newVar8) {
        this->dynamicVar8 = newVar8;
    }

    void setDynamicVar16(uint16_t newVar16) {
        this->dynamicVar16 = newVar16;
    }

protected:
    static uint8_t staticVar8;
    static uint16_t staticVar16;
    uint8_t dynamicVar8;
    uint16_t dynamicVar16;
};

uint8_t BaseClass::staticVar8 = 0;
uint16_t BaseClass::staticVar16 = 0;

BaseClass::BaseClass() {
    printf("BaseClass::BaseClass()\n");
    this->staticVar8 = 34;
    this->staticVar16 = 196;
    this->dynamicVar8 = 35;
    this->dynamicVar16 = 197;
}

class DerivedClass1 : public BaseClass {
public:
    DerivedClass1(int);
    void runThread();
private:
    uint8_t myVar8;
    uint16_t myVar16;
};

class DerivedClass2 : public BaseClass {
public:
    DerivedClass2(int);
    void runThread();
private:
    uint8_t myVar8;
    uint16_t myVar16;
};



DerivedClass1::DerivedClass1(int foo) {
    printf("DerivedClass1::DerivedClass1(%d)\n", foo);
    this->myVar8 = foo;
    this->myVar16 = foo * 2;
}

DerivedClass2::DerivedClass2(int foo) {
    printf("DerivedClass2::DerivedClass2(%d)\n", foo);
    this->myVar8 = foo;
    this->myVar16 = foo * 2;
}

void DerivedClass1::runThread() {
    printf("Derived Class 1\n");
    printf("staticVar8: %d\n", this->staticVar8);
    printf("staticVar16: %d\n", this->staticVar16);
    printf("dynamicVar8: %d\n", this->dynamicVar8);
    printf("dynamicVar16: %d\n", this->dynamicVar16);
    printf("myVar8: %d\n", this->myVar8);
    printf("myVar16: %d\n", this->myVar16);
}

void DerivedClass2::runThread() {
    printf("Derived Class 2\n");
    printf("staticVar8: %d\n", this->staticVar8);
    printf("staticVar16: %d\n", this->staticVar16);
    printf("dynamicVar8: %d\n", this->dynamicVar8);
    printf("dynamicVar16: %d\n", this->dynamicVar16);
    printf("myVar8: %d\n", this->myVar8);
    printf("myVar16: %d\n", this->myVar16);
}

int main() {
    std::vector<BaseClass*> ClassList;
    int i;

    BaseClass *myBaseClass;

    myBaseClass = new DerivedClass1(8);
    ClassList.push_back(myBaseClass);
    myBaseClass = new DerivedClass1(12);
    ClassList.push_back(myBaseClass);
    myBaseClass = new DerivedClass2(16);
    ClassList.push_back(myBaseClass);
    myBaseClass = new DerivedClass2(20);
    ClassList.push_back(myBaseClass);

    for (i = 0; i < ClassList.size(); i++) {
        printf("\n\nEntry %d: \n", i);
        ClassList[i]->runThread();
    }

    for (i = 0; i < ClassList.size(); i++) {
        ClassList[i]->setStaticVar8(i);
        ClassList[i]->setStaticVar16(i);
        ClassList[i]->setDynamicVar8(i);
        ClassList[i]->setDynamicVar16(i);
    }

    for (i = 0; i < ClassList.size(); i++) {
        printf("\n\nEntry %d: \n", i);
        ClassList[i]->runThread();
    }

}