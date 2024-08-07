omp_v1.c：直接套用pragma omp parallel for

omp_v2.c：循环展开

omp_v3.c：CSRL策略

omp_v4.c：重新划分任务达到负载均衡

omp_v5.c：并行区开在外面