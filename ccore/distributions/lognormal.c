#include "distributions.h"
#include <math.h>
#include <stdlib.h>

double log_normal_sample(double mu, double sigma) {
    return exp(normal_sample(mu, sigma));
}