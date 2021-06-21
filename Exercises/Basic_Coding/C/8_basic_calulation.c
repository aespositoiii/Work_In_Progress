#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(){
    
    double num1;
    double num2;

    printf("Number One: ");
    scanf("%lf", &num1);
    printf("Number Two: ");
    scanf("%lf", &num2);
    printf("Number One (%lf) plus Number 2 (%lf) is %f\n", num1, num2, num1+num2);


    return 0;
}