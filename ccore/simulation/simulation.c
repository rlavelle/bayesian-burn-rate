#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../distributions/distributions.h"
#include "../samplers/samplers.h"
#include "simulation.h"

int random_index(int size) {
    return (int) uniform_sample(0, size);
}

simulation_t *simulate_spending(double funds,
                                double (*change_func)(double),
                                log_norm_samp_t *gibbs_dist,
                                double gibbs_dist_size,
                                int n_iters) {
    simulation_t *res = (simulation_t*)malloc(n_iters*sizeof(simulation_t*));

    for(int i = 0; i < n_iters; i++){
        int capacity = 10;
        double *tmp_data = malloc(capacity*sizeof(double));
        double funds_copy = funds;

        int flag = 0;
        int j = 0;
        tmp_data[j++] = funds_copy;
        while(funds_copy > 0) {
            if(j == capacity) {
                capacity *= 2;
                double *tmp = realloc(tmp_data, capacity*sizeof(double));
                tmp_data = tmp;
            }

            double sample = gibbs_dist[random_index(gibbs_dist_size)].ypred;
            if(uniform_sample(0,1) < change_func(j) || flag){
                sample += 2300;
                flag = 1;
            }

            funds_copy -= sample;
            tmp_data[j++] = funds_copy;
        }

        simulation_t *sim_data = (simulation_t*)malloc(sizeof(simulation_t));
        sim_data->sim = malloc(j*sizeof(double));
        sim_data->sim_len = j;

        for(int k = 0; k<j; k++){
            sim_data->sim[k] = *(tmp_data + k);
        }

        free(tmp_data);

        *(res + i) = *sim_data;
    }

    return res;
}