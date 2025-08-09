#include <stdio.h>
#include "distributions/distributions.h"
#include <math.h>

int main() {
    int N = 1000000;
    double rngs[N];
    double sum = 0.0;

    for(int i = 0; i<N; i++){
        rngs[i] = normal_sample(0.0,1.0);
        sum += rngs[i];
    }

    double mu = sum/N;

    double var = 0.0;
    for(int i = 0; i<N; i++){
        var += pow(rngs[i] - mu, 2);
    }

    var = var/N;

    printf("mu=%f var=%f\n", mu, var);

    
    return 0;
}