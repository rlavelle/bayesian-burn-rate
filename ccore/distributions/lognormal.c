#include "distributions.h"
#include <math.h>
#include <stdlib.h>

// use fact that log(X) ~ N(mu,sigma)
// X ~ exp(N(mu, sigma))
double log_normal_sample(double mu, double sigma) {
    return exp(normal_sample(mu, sigma));
}