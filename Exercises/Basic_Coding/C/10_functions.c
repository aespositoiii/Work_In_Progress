#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void sayHi(char name[]){
    printf("'Ello %s ;-x", name);
}

int main(){
    
    char name[20];
    printf("Whoayou?\n");
    scanf("%s", name);
    printf("Top\n");
    sayHi(name);
    printf("\nBottom\n");

    return 0;
}   

