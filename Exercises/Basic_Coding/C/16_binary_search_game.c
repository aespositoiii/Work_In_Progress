#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

int random_number(int max){
    
    time_t t;

    srand((unsigned) time(&t));
    
    int rand_num = (rand() % max) + 1;



    return rand_num;
}

int main(){

    int range_max;

    printf("\n\nHow high do you want to guess?\n");
    scanf("%d", &range_max);

    int secret_number = random_number(range_max);
    //printf("%d", secret_number);

    printf("\n\nOK, Start Guessing!\n");

    int guesses = 0;
    int guess;

    while( guess != secret_number ){
        guesses++;
        scanf("%d", &guess);
        
        if( guess > secret_number){
            printf("\nToo High! Try a lower number!  ");
        } else if( guess < secret_number ){
            printf("\nToo Low! Try a higher number!  ");
        }
    }

    printf("\nYou got it! The secret number was %d!\nIt took you %d guesses to find the answer.\n\n", secret_number, guesses);

    return 0;
}   
