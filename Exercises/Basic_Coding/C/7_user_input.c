#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(){
    
    int num;
    char letter;
    char word[20];
    char full_name[30];

    printf("Tell me an integer!\n");

    scanf("%d", &num); // '&' is a pointer

    printf("\nYour number is %d.\n", num);

    printf("Tell me a letter!\n");

    scanf(" %c", &letter);    

    printf("\nYour letter is %c\n", letter);

    printf("What is the word?\n");

    scanf("%s", word);

    printf("%s %s %s, %s is the word.\n", word, word, word, word);

    printf("Tell me a letter!\n");

    scanf(" %c", &letter);    

    printf("\nYour letter is %c\n", letter);

    printf("\n TELL ME YOUR FIRST AND LAST NAME!\n");

    fgets(full_name, 30, stdin);

    printf("%s is your full name.\n\n", full_name);


    return 0;
}