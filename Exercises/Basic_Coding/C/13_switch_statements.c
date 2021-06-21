#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(){

    char grade;

    printf("\n\nWhat grade did you get?\n");
    scanf(" %c", &grade);

    switch(grade){
        case 'A' :
            printf("\n\nNice, you got the best grade possible!\n\n");
            break;
        case 'B' :
            printf("\n\nNot bad, keep up the good work.\n\n");
            break;
        case 'C' :
            printf("\n\nYou passed but try to do better next time.\n\n");
            break;
        case 'D' :
            printf("\n\nThat's not so good, do you know where you went wrong?\n\n");
            break;
        case 'F' :
            printf("\n\nYou should ask someone for help.  That's as bad as it gets.\n\n");
            break;
        default :
            printf("\n\n...that's not a valid grade....\n\n");
            break;
    }

    return 0;
}   
