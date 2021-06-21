#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(){
    
    int a = 10;
    int b = 25;
    int c = a + b;
    float d = b;
    float e = d / a;
    double g = 5;
    double h = 2;

    printf("a = %d\n", a);
    printf("b = %d\n", b);
    printf("d = b = %f\n", d);
    printf("c = a + b =  %d\n", c);
    printf("e = d / a =  %f\n", e);
    printf("f = a^2 = %f", pow(g, h));


    return 0;
}