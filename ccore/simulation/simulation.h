#ifndef SIMULATION_H
#define SIMULATION_H
#define CHANGE_POINT_MONTH 10 
#include "../samplers/samplers.h"

typedef struct {
    double *sim;
    int sim_len;
} simulation_t;

simulation_t *simulate_spending(double funds,
                                double (*change_func)(double),
                                log_norm_samp_t *gibbs_dist,                                
                                double gibbs_dist_size,
                                int n_iters);

double change_point_func(double x, double x0, double k, double m, double L, double Q);

#endif