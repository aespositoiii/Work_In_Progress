#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

struct Car{
    char make[20];
    char model[20];
    char trim[20];
    int year;
    int price;
};

int main(){

    struct Car car1;

    strcpy( car1.make, "Subaru");
    strcpy( car1.model, "Outback");
    strcpy( car1.trim, "2.5i_Limited"); 
    car1.year = 2012;
    car1.price = 13995;

    printf("\n\nMake : %s\nModel : %s\nTrim : %s\nYear : %d\nPrice : $%d.00\n\n", car1.make, car1.model, car1.trim, car1.year, car1.price);

    return 0;
}   
