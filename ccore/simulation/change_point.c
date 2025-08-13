#include <math.h>
#include <stdlib.h> 
#include "simulation.h"

double sigmoid(double x, double x0, double k, double m, double L) {
    return L / (1.0 + exp(-k * (x - x0))) - m;
}

double change_point_func(double x, double x0, double k, double m, double L, double Q) {
    double result;
    if (x < Q) {
        if (x < Q) {
            result = 0.05 * (x / Q);
        } else {
            result = 1.0;
        }
    } else {
        result = sigmoid(x, x0, k, m, L);
    }
    
    return result;
}