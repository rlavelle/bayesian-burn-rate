#include "distributions.h"
#include <math.h>
#include <stdlib.h>

// box-muller transform
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
        x = uniform_sample(-1,1); 
        y = uniform_sample(-1,1);
        r = x*x + y*y; // euclid dist
    } while(r == 0.0 || r > 1.0); // outside unit circle

    double d = sqrt(-2.0 * log(r)/r);
    spare = y*d;

    return mu + sigma*x*d;
}