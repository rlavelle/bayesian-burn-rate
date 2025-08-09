#include "distributions.sh"
#include <math.h>
#include <stdlib.h>

double normal_sample(double mu, double sigma) {
    static int has_spare = 0;
    static double spare;

    if(has_spare) {
        has_spare = 0;
        return mu + spare*sigma;
    }

    has_spare = 1;
    double x,y,r;

    do {
        // ~ U(-1,1)
        x = 2*rand()/RAND_MAX - 1; 
        y = 2*rand()/RAND_MAX - 1;
        r = x*x + y*y // euclid dist
    } while(r == 0.0 || r > 1.0); // outside unit circle

    double d = sqrt(-2.0 * log(r)/r);
    spare = y*d;

    return mu + sigma*x*d;
}

double normal_pdf(double x, double mu, double sigma) {
    // (1/sqrt(sigma^2*2*pi)*exp(-(x-mu)^2 / (2*sigma^2)))
    double exponent = -(x - mu) * (x - mu) / (2 * sigma * sigma);
    return (1.0 / (sigma * sqrt(2 * M_PI))) * exp(exponent);
}