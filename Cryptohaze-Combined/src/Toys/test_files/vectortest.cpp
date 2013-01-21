#include <stdio.h>
#include <vector>



int main() {
    int i;
    std::vector<int> test;

    printf("Vector testing!\n");

    for (i = 0; i < 100; i++) {
        test.push_back(i);
    }

    printf("Vector readout: \n");
    for (i = 0; i < 100; i++) {
        printf("test[%d]: %d\n", i, test.at(i));
    }

    printf("Vector readout: \n");
    for (i = 0; i < 100; i++) {
        printf("test[%d]: %d\n", i, test.at(i));
    }
}