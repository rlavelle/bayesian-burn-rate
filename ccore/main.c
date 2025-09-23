#include <stdio.h>
#include <stdlib.h> 
#include "distributions/distributions.h"
#include "samplers/samplers.h"
#include "simulation/simulation.h"
#include "core.h"
#include <math.h>
#include "string.h"

int main() {
    char data_path[256] = "/Users/rowanlavelle/Documents/Projects/bayesian-burn-rate/data/spend.txt";
    data_t *spend = read_data(data_path);

    char funds_path[256] = "/Users/rowanlavelle/Documents/Projects/bayesian-burn-rate/data/funds.txt";
    data_t *funds = read_data(funds_path);

    change_point_params_t params = {1,0.5,1.5};
    
    for(int i = 3; i<spend->n_data; i++){
        int n_data = i; //spend->n_data;

        double *data = malloc(sizeof(double)*i);
        for(int j = 0; j<i; j++){
            data[j] = spend->data[j];
        }
        int n_iter = 100000;
        
        log_norm_priors_t *priors = &DEFAULT_PRIORS;
        log_norm_samp_t *samples = gibbs_sampler(priors, data, n_data, n_iter);

        char out_path[256];
        char num_str[16]; 

        snprintf(num_str, sizeof(num_str), "%d", i);
        strcpy(out_path, "/Users/rowanlavelle/Documents/Projects/bayesian-burn-rate/data/gibbs_results_");
        strcat(out_path, num_str);
        strcat(out_path, ".csv");
        write_sampler_results(out_path, samples, n_iter);
        
        double sim_iters = 100000;
        double Q = 17.0-i; // 16 months from first run is change point, 15 from 2nd,...

        simulation_t *simulation_results = simulate_spending(
            funds->data[i],change_point_func,params,Q,samples,n_iter,sim_iters
        );
        
        char out_path2[256];
        strcpy(out_path2, "/Users/rowanlavelle/Documents/Projects/bayesian-burn-rate/data/sim_results_");
        strcat(out_path2, num_str);
        strcat(out_path2, ".csv");

        write_simulation_results(out_path2, simulation_results, sim_iters);
    }

    return 0;
}