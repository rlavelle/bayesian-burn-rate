#include <math.h>
#include <stdlib.h> 
#include "simulation.h"

double sigmoid(double x, double x0, double k, double m, double L) {
    return L / (1.0 + exp(-k * (x - x0))) - m;
}

/*
Defaults:
k=1, m=0, L=1.5
*/
double change_point_func(double x, double k, double m, double L, double Q) {
    double result;
    if (x < Q) {
         // same spending if x < Q, small chance of rent
        result = 0.02;
    } else {
        // after month Q, sigmoid func to describe chance in rent
        result = sigmoid(x, Q, k, m, L);
    }
    
    return result;
}