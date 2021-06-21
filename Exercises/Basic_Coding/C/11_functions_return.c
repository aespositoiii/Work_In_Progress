#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double cube(double num);  // prototyping allows you to give the program function prototypes before the main function.  The compiler gets the body from somehwere else.

int main(){

    double number;
    printf("\n\nWhat number do you want to cube?  ");
    scanf("%lf", &number);
    printf("\n\nThe cube of %f is %f.\n\n", number, cube(number));

    return 0;
}   

double cube(double num){
    double result = num * num * num;
    return result;
}