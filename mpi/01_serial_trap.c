/*
 * File:        01_serial_trap.c
 * Purpose:     Calculate area by using trapezoidal rule
 * Compile:     gcc -Wall -o 01_serial_trap 01_serial_trap.c
 * Run:
 *  01_serial_trap <a> <b> <n>
 *      - a: left end-point
 *      - b: right end-point
 *      - n: the number of subinterval
*/
#include <stdio.h>
#include <stdlib.h>

double f(double x);
double Trap(double a, double b, int n, double h);

int main(int argc, char** argv)
{
    double integral;
    double a, b;
    int n;
    double h;

    if (argc != 4) {
        fprintf(stderr, "usage: %s <a> <b> <n>\n", argv[0]);
        fprintf(stderr, "   a: left end-point\n");
        fprintf(stderr, "   b: right end-point\n");
        fprintf(stderr, "   n: the number of subinterval\n");
        exit(-1);
    }

    a = atof(argv[1]);
    b = atof(argv[2]);
    n = atoi(argv[3]);

    h = (b-a)/n;
    integral = Trap(a, b, n, h);

    printf("With n = %d trapezoids, our estimate\n", n);
    printf("of the integral from %f to %f = %.15f\n", a, b, integral);

    return 0;
}

double Trap(double a, double b, int n, double h)
{
    double integral;
    
    integral = (f(a) + f(b)) / 2.0;

    for(int k = 0; k < n; k++) {
        integral += f(a + k*h);
    }
    integral = integral * h;

    return integral;
}

double f(double x)
{
    return x*x;
}