#include "distributions.h"
#include <math.h>
#include <stdlib.h>

// rejection sampling from George Marsaglia and Wai Wan Tsang, et al
// impl: https://daannoordenbos.github.io/gamma-sampling/
double gamma_sample (double alpha, double beta) {
    double d, c, x, xs, v, u;
    if (alpha >= 1.0) {
        d = alpha - 1.0 / 3.0;
        c = 1.0 / sqrt(9.0 * d);
        while (1) {
            v = -1;
            while (v <= 0) {
                x = normal_sample(0, 1);
                v = 1.0 + c * x;
            }
            v = v * v * v;
            u = uniform_sample(0,1);
            xs = x * x;
            if (u < 1.0 - 0.0331 * xs * xs || log(u) < 0.5 * xs + d * (1.0 - v + log(v)))
                return d * v / beta;
        }
    }
    else{
        double g = gamma_sample(alpha + 1.0, 1.0);
        double w = uniform_sample(0, 1);
        return g * pow(w, 1.0 / alpha) / beta;
    }
}
