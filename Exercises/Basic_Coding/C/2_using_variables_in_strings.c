#include <stdio.h>
#include <stdlib.h>

int main(){
    
    char pocket_contents[] = "colostomy bags";
    int contents_count = 15;

    printf("\nHey, this guy has a pocket full of %s!\n", pocket_contents);
    printf("He's got like %d %s in his pocket!\n\n", contents_count, pocket_contents);
    
    return 0;
}