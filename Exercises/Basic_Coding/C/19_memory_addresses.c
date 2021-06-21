#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>


int main(){

    int i;

    for( i = 0 ; i <10 ; i++){
        printf("%d : %p\n", i , &i);
    }

    return 0;
}   
