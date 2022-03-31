// C program to read a BMP Image and 
// write the same into a PGM Image file
#include <stdio.h>
  
void main()
{
    int i, j, temp = 0;
    int width = 1024, height = 1024;
  
    FILE* pgmimg;
    pgmimg = fopen("pgmimg.pgm", "wb");
  
    // Writing Magic Number to the File
    fprintf(pgmimg, "P2\n"); 
  
    fprintf(pgmimg, "# Written by pgmwrite\n");
    // Writing Width and Height
    fprintf(pgmimg, "%d %d\n", width, height); 
  
    // Writing the maximum gray value
    fprintf(pgmimg, "255\n"); 
    int count = 0;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            temp = 50;
            // Writing the gray values in the 2D array to the file
            fprintf(pgmimg, "%d ", temp);
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}