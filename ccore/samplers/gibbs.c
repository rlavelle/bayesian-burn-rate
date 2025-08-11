// assume data y1,...,yn is log normal with unknown theta and sigma^2
// let x1,...,xn = log(y1),...,log(yn)
// p(theta | sigma^2, X) /propto p(x|theta,sigma^2)p(theta)
//     we let p(theta) ~ N(mu_0, k^2_0) be our prior 
//     and p(X|theta,sigma^2) ~ prod[Norm(x_i|theta,sigma^2)] be the likelihood
// which gives the posterior theta ~ N(mu_n, k^2_n) where 
//     mu_n = [mu_0/k^2_0 + n*(xbar/sigma^2)] / [1/k^2_0 + n/sigma^2]
//     k^2_n = [1/k^2_0 + n/sigma^2]^(-1)
// this relies on xbar (sample mean) and sigma^2
// p(sigma^2 | theta, X) \propto p(x|theta,sigma^2)p(sigma^2)
//     we let p(sigma^2) ~ Gamma(v_0/2,v_0*sigma^2_0/2) be our prior
// which gives the posterior sigma^2 ~ Gamma(v_n/2, v_n*sigma^2_n/2) where
//     v_n = v_0 + n
//     sigma^2_n = (1/v_n)[v_0*sigma^2_0 + n*s^2] 
// where s^2 is the sample variance of the data (assuming theta as mean)
// we can then use a gibbs sampler to draw samples of our parameters
// 1) sample theta_(i) ~ N(mu_n^(i-1), k^2_n^(i-1))
// 2) sample 1/sigma^2_(i) ~ Gamma(v_n/2, v_n*sigma^2_n/2^(i-1))
// 3) save {theta_(i), sigma^2_(i)} as sample parameters from the gibbs sampler
// 4) y_(i) ~ LogNorm(theta_(i), sigma^2_(i))
// (4) follows as x_(i) ~ N(theta_(i), sigma^2_(i)) where x_i = log(y_i)
//     and LogNorm(mu,sigma) = exp(N(mu,sigma))

#include "../distributions/distributions.h"
#include "samplers.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double compute_mean(double *y, int n_data) {
    double sum = 0;
    for(int i = 0; i<n_data; i++){
        sum += *(y+i);
    }

    return sum / n_data;
}

double compute_ss(double theta, double *y, int n_data) {
    double ss = 0;
    for(int i = 0; i<n_data; i++){
        ss += pow(theta - *(y+i),2);
    }

    return ss / n_data;
}

log_norm_samp *gibbs_sampler(log_norm_priors *priors, double *y, int n_data, int n_iter) {
    log_norm_samp *result = (log_norm_samp*)malloc(sizeof(log_norm_samp)*n_iter);
    
    // let x1,...,xn = log(y1),...,log(yn)
    double *x = (double*)malloc(sizeof(double)*n_data);
    for(int i = 0; i<n_data; i++){
        *(x+i) = log(*(y+i));
    }

    double xbar = compute_mean(x, n_data);

    int i = 0;
    // need an init sigma2_i value to sample theta_i, use mean of gamma prior (beta / (alpha-1))
    double sigma2_i =  1 / (priors->gamma_prior.sigma20 / (priors->gamma_prior.v0 - 1));
    double theta_i;

    while(i < n_iter) {
        // 1) sample theta_(i) ~ N(mu_n^(i-1), k^2_n^(i-1))   
        // mu_n = [mu_0/k^2_0 + n*(xbar/sigma^2)] / [1/k^2_0 + n/sigma^2]
        // k^2_n = [1/k^2_0 + n/sigma^2]^(-1)  
        double mu_n = (priors->norm_prior.mu0/priors->norm_prior.k20 + n_data*(xbar/sigma2_i)) / 
                      (1/priors->norm_prior.k20 + n_data/sigma2_i);
        double k2n = 1 / (1/priors->norm_prior.k20 + n_data/sigma2_i);
        theta_i = normal_sample(mu_n, sqrt(k2n));

        // 2) sample 1/sigma^2_(i) ~ Gamma(v_n/2, v_n*sigma^2_n/2^(i-1))
        // v_n = v_0 + n
        // sigma^2_n = (1/v_n)[v_0*sigma^2_0 + n*s^2]   
        double vn = priors->gamma_prior.v0 + n_data;
        double s2 = compute_ss(theta_i, x, n_data);
        double sigma2n = (1/vn)*(priors->gamma_prior.v0*priors->gamma_prior.sigma20 * n_data*s2);
        sigma2_i = 1/gamma_sample(vn/2, vn*sigma2n/2);

        // 3) save {theta_(i), sigma^2_(i)} as sample parameters from the gibbs sampler
        // 4) y_(i) ~ LogNorm(theta_(i), sigma^2_(i))
        double ypred = log_normal_sample(theta_i, sqrt(sigma2_i));

        log_norm_samp sample = {
            .params = {
                .theta = theta_i,
                .sigma2 = sigma2_i
            },
            ypred = ypred
        };

        *(result + i) = sample;

        i++;

        if(i%10000 == 0){
            printf("%d iters complete...\n", i);
        }
    }   

    return result;

}
