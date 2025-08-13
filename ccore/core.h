#ifndef CORE_H
#define CORE_H
#include "samplers/samplers.h"
#include "simulation/simulation.h"

typedef struct {
    double *data;
    int n_data;
} data_t;

data_t *read_data(const char *fpath);
int write_sampler_results(const char* fpath, log_norm_samp_t *samples, int size);
int write_simulation_results(const char* fpath, simulation_t *sims, int size);

#endif