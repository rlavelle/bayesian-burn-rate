#ifndef SAMPLERS_H
#define SAMPLERS_H

typedef struct {
    double mu0;
    double k20;
} norm_prior;

typedef struct {
    double v0;
    double sigma20;
} gamma_prior;

typedef struct {
    norm_prior norm_prior;
    gamma_prior gamma_prior;
} log_norm_priors;

typedef struct {
    double theta;
    double sigma2;
} log_norm_params;

typedef struct {
    log_norm_params params;
    double ypred;
} log_norm_samp;

static const log_norm_priors DEFAULT_PRIORS = {
    .norm_prior = {
        .mu0 = 7.82, 
        .k20 = 0.001
    },

    .gamma_prior = {
        .v0 = 2,
        .sigma20 = 0.5
    }
};

log_norm_samp *gibbs_sampler(log_norm_priors *priors, double *y, int n_data, int n_iter);

#endif 