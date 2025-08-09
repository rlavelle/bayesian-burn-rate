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
 // we can then use a gibbs sample to draw samples of our data
 // 1) sample 1/sigma^2_(i) ~ Gamma(v_n/2, v_n*sigma^2_n/2)
 // 2) sample theta_(i) ~ N(mu_n, k^2_n)
 // 3) save {theta_(i), sigma^2_(i)} as sample parameters from the gibbs sampler
 // 4) y_(i) ~ LogNorm(theta_(i), sigma^2_(i))
 // (4) follows as x_(i) ~ N(theta_(i), sigma^2_(i)) where x_i = log(y_i)
 //     and LogNorm(mu,sigma) = exp(N(mu,sigma))

 #include "../distributions/distributions.h"
