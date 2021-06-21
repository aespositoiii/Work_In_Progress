#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int max2(int num1, int num2){
    int result;
    if(num1 > num2){
        result = num1;
    } else {
        result = num2;
    }
    return result;
}

int max3(int num1, int num2, int num3){
    int result;
    if(num1 >= num2 && num1 >= num3){
        result = num1;
    } else if(num2 >= num1 && num2 >= num3){
        result = num2;
    } else {
        result = num3;
    }
    return result;
}

int main(){
    int first;
    int second;
    int third;
    printf("\n\nWhat is the first number?\n");
    scanf("%d", &first);
    printf("\n\nWhat is the second number?\n");
    scanf("%d", &second);
    printf("\n\nWhat is the third number?\n");
    scanf("%d", &third);
    printf("\n\nThe greatest number is %d. \n\n", max3(first, second, third));


    return 0;
}   
