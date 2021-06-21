#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


int main(){

    int num[] = {1,2,3,5,8,13,21,34,55,89};
    int size = sizeof num / sizeof num[0];

    int i = 0;

    while ( i < size){
        printf("Number %d is %d.\n", i+1, num[i]);
        i++;
    }

    i = 10;

    printf("\n\nDo While i < 10\n");
    do{ 
        printf("%d\n", i);
        i++;
    }while(i < 10);

    i = 10;

    printf("\n\nWhile i < 10\n");
    while(i < 10){
        printf("%d\n", i);
        i++;
    }

    return 0;
}   
