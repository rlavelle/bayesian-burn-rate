#ifndef SAMPLERS_H
#define SAMPLERS_H

typedef struct {
    double mu0;
    double k20;
} norm_prior_t;

typedef struct {
    double v0;
    double sigma20;
} gamma_prior_t;

typedef struct {
    norm_prior_t norm_prior;
    gamma_prior_t gamma_prior;
} log_norm_priors_t;

typedef struct {
    double theta;
    double sigma2;
} log_norm_params_t;

typedef struct {
    log_norm_params_t params;
    double ypred;
} log_norm_samp_t;

static const log_norm_priors_t DEFAULT_PRIORS = {
    .norm_prior = {
        .mu0 = 7.82, 
        .k20 = 0.001
    },

    .gamma_prior = {
        .v0 = 2,
        .sigma20 = 0.5
    }
};

log_norm_samp_t *gibbs_sampler(log_norm_priors_t *priors, double *y, int n_data, int n_iter);

#endif 