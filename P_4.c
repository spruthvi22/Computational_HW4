/*
* Author:Pruthvi Suryadevara
* Email: pruthvi.suryadevara@tifr.res.in
* Description: Random numbers using Transformation method
* Compile using gcc P_4.c -lm -o P_4.out
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void)
{
  int n = 10000;
  double *y_cons, *y_exp, *h_cons, *h_exp;
  y_cons = (double*) malloc(sizeof(double) * n);
  y_exp = (double*) malloc(sizeof(double) * n);
  h_cons = (double*) malloc(sizeof(double) * 10);
  h_exp = (double*) malloc(sizeof(double) * 10);

  for(int i; i<n; i++)
    {
      y_cons[i] = (double)rand()/(double)RAND_MAX;  // Creating rendom number between 0,1
      y_exp[i] = -2 * log(y_cons[i]);               // Using transformation to get exponential distribution
    }
  

  FILE *fp;      // Saving to csv file
  int st=remove("P_4.csv");
  fp = fopen("P_4.csv", "w+");
  for(int i=0;i<n;i++)
    {
      fprintf(fp, "%f,%f \n", y_cons[i], y_exp[i]);
    }
  
  
  return(0);
}
