#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

double uniform_sample(double a, double b);
double normal_sample(double mu, double sigma);
double log_normal_sample(double mu, double sigma);
double gamma_sample(double alpha, double beta);

#endif