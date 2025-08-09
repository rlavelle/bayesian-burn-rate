#include "distributions.h"
#include <math.h>
#include <stdlib.h>

double uniform_sample(double a, double b) {
    double U = rand()/((double)RAND_MAX);
    return a + (b-a)*U
}