#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>


int main(){

    int i;
    int *pI;

    for( i = 0 ; i <10 ; i++){
        pI = &i;
        printf("%d : %p : %d\n", i, pI , *pI);
    }

    return 0;
}   
