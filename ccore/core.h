#ifndef CORE_H
#define CORE_H
#include "samplers/samplers.h"

typedef struct {
    double *data;
    int n_data;
} spend_data;

spend_data *read_data(const char *fpath);
int write_sampler_results(const char* fpath, log_norm_samp *samples, int size);

#endif