#include <stdio.h>
#include "distributions/distributions.h"
#include "samplers/samplers.h"
#include "core.h"
#include <math.h>

int main() {
    char data_path[256] = "/Users/rowanlavelle/Documents/Projects/bayesian-burn-rate/data/spend.txt";
    spend_data *spend = read_data(data_path);
    int n_data = spend->n_data;
    double *data = spend->data;

    int n_iter = 100000;
    
    log_norm_priors *priors = &DEFAULT_PRIORS;
    log_norm_samp* samples = gibbs_sampler(
        priors, data, n_data, n_iter
    );

    char out_path[256] = "/Users/rowanlavelle/Documents/Projects/bayesian-burn-rate/data/results.csv";
    write_sampler_results(out_path, samples, n_iter);

    return 0;
}