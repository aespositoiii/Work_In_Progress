#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>


int main(){

    FILE * fpointer = fopen("employees.txt", "a");
    
    fprintf(fpointer, "Jim Beam - Handyman\nTom Collins - Carpenter\nMabel Barker - Homemaker\n");

    fclose(fpointer);

    return 0;
}   
