#include <stdio.h>
#include <stdlib.h>
#include "core.h"
#include "samplers/samplers.h"

spend_data *read_data(const char *fpath) {
    FILE *file = fopen(fpath, "r");
    if(file == NULL){
        printf("failed to open file\n");
        return NULL;
    }

    int *tmp_data = NULL;
    int capacity = 10;
    int count = 0;

    tmp_data = malloc(capacity*sizeof(int));

    int num;
    while(fscanf(file, "%d", &num) == 1) {
        if(count == capacity){
            capacity *= 2;
            int *tmp = realloc(tmp_data, capacity*sizeof(int));
            tmp_data = tmp;
        }
        tmp_data[count++] = num;
    }

    if(!feof(file)) {
        printf("error reading file (not all values stored)");

    }

    spend_data *spend = (spend_data*)malloc(sizeof(spend_data));
    spend->data = malloc(count*sizeof(double));
    spend->n_data = count;
    
    for(int i = 0; i<count; i++){
        spend->data[i] = (double)*(tmp_data + i);
    }

    free(tmp_data);
    fclose(file);

    return spend;
}

int write_sampler_results(const char* fpath, log_norm_samp *samples, int size){
    FILE* file = fopen(fpath, "w");
    if (file == NULL) {
        perror("Error opening file");
        return -1;
    }

    fprintf(file, "theta,sigma2,ypred\n");
    for (int i = 0; i < size; i++) {
        double theta = samples[i].params.theta;
        double sigma2 = samples[i].params.sigma2;
        double ypred = samples[i].ypred;

        if (fprintf(file, "%f,%f,%f\n", theta, sigma2, ypred) < 0) {
            perror("Error writing to file");
            fclose(file);
            return -1;
        }
    }

    if (fclose(file) != 0) {
        perror("Error closing file");
        return -1;
    }

    printf("Successfully wrote %d elements to %s\n", size, fpath);
    return 0;
}

