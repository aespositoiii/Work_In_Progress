#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>


int main(){

    int nums[4][3] = {
                        { 1, 2, 3},
                        { 4, 5, 6}
                    };

    int size = sizeof nums;
    int sizen = sizeof nums / sizeof nums[0];
    int sizem = size / (sizen * sizeof nums[0][0]);

    /*
    printf("\n%d\n\n", size);
    printf("\n%d\n\n", sizen);
    printf("\n%d\n\n", sizem);
    */

    int i, j;
    for( i = 0 ; i < sizen ; i++ ){
        
        printf("\n  ");

        for( j = 0 ; j < sizem ; j++ ){
            
            printf("  '%d'  ", nums[i][j]);
            
            }
        printf("\n");
    }
    return 0;
}   
