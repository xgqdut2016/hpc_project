#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#define max_function(a,b) ((a)>(b)?(a):(b))

void read_non(const char *filename, int *num){
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }
    
    fscanf(file, "%d %d %d\n", &num[0] , &num[1] , &num[2]);// 只能读取指针
    fclose(file);
}

void read_matrix(const char *filename, int *num, int *row_ptr_t, int *col_ind_t, double *values_t){
    int num_rows, num_vals;
    int i;
    num_rows = num[0];
    num_vals = num[2];
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }

    //统计每一行非零元个数--------------------------------------------
    int row_occurances[num_rows];
    
    for(i = 0; i < num_rows; i++){
        row_occurances[i] = 0;// init data
    }
    int row, column;
    double value;
    while (fscanf(file, "%d %d %lf\n", &row, &column, &value) != EOF) {
        // Subtract 1 from row and column indices to match C format
        row--;//非零元的行索引，必须要减一
        column--;
        
        row_occurances[row]++;//此时第row个元素存储的是该row的非零元个数
    }
    
    
    //使用CSR格式读取矩阵元素---------------------------------
    // Set row offset
    int index = 0;
    for (i = 0; i < num_rows; i++) {
        row_ptr_t[i] = index;//这个才是那个row offset数组
        index += row_occurances[i];
    }//row_ptr_t[i + 1] - row_ptr_t[i]表示第i行元素个数
    row_ptr_t[num_rows] = num_vals;//最后一个元素就是非零元总数
    
    
    //根据row offset 读取列索引
    // Set the file position to the beginning of the file
    rewind(file);
    for(i = 0; i < num_vals; i++){
        col_ind_t[i] = -1;//init data
    }
    fscanf(file, "%d %d %d\n", &num[0] , &num[1] , &num[2]);//不可删除
    i = 0;
    while (fscanf(file, "%d %d %lf\n", &row, &column, &value) != EOF) {
        row--;
        column--;
        
        // Find the correct index (i + row_ptr_t[row]) using both row information and an index i
        while (col_ind_t[i + row_ptr_t[row]] != -1) {//col_ind_t[row_ptr_t[row]:row_ptr_t[row + 1]]表示第row非零元的列索引
            i++;
        }//i + row_ptr_t[row]表示第row行一共有row_ptr_t[row]个非零元，
        col_ind_t[i + row_ptr_t[row]] = column;
        values_t[i + row_ptr_t[row]] = value;
        i = 0;
    }
    
    fclose(file);

}

void spmv_csr(const int *row_ptr, const int *col_ind, const double *values, const int num_rows,const int *row_task, const double *x, double *y) {
    int i;
    int tid = omp_get_thread_num();
    for (i = row_task[tid]; i < row_task[tid + 1]; i++) {
        double sum = 0;
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        
        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * x[col_ind[j]];
        }
        
        y[i] = sum;
    }

}

int main(int argc, const char * argv[]) {
    
    int num_repeat = 500;
    int row_task[9] = {0,509,1000,1470,1937,2417,2904,3384,3876};
    const char *filename = "./input.mtx";
    //-----------------------
    int num[3],nthreads;
    read_non(filename, num);
    int num_rows, num_vals;
    num_rows = num[0];
    num_vals = num[2];
    
    int row_ptr_t[num_rows + 1];
    int col_ind_t[num_vals];
    double values_t[num_vals];
    read_matrix(filename, num, row_ptr_t, col_ind_t, values_t);
    //init x and y
    int i;
    double *x = (double *)malloc(num_rows * sizeof(double)); 
    double *y = (double *)malloc(num_rows * sizeof(double)); 
    double *p_x = (double *)malloc(num_rows * sizeof(double)); 
    double *p_y = (double *)malloc(num_rows * sizeof(double));
    for(i = 0; i < num_rows; i++){
        x[i] = 1.0;
        y[i] = 0.0;
        p_x[i] = 1.0;
        p_y[i] = 0.0;
    }

    for(int j = 0; j < num_repeat; j++){
        #pragma omp parallel num_threads(1) default(shared)
        {
            for (i = 0; i < num_rows; i++) {
                double sum = 0;
                const int row_start = row_ptr_t[i];
                const int row_end = row_ptr_t[i + 1];
        
                for (int j = row_start; j < row_end; j++) {
                    sum += values_t[j] * x[col_ind_t[j]];
                }
                y[i] = sum;
            }
            #pragma omp barrier
            #pragma omp for simd private(i) schedule(static)
            for(i = 0; i < num_rows; i++){
                x[i] = y[i]/1e2 + 1.0;
                y[i] = 0.0;
            }
        }  
    }

    double start,ela;

    start = omp_get_wtime();
    for(int j = 0; j < num_repeat; j++){
        #pragma omp parallel num_threads(8) default(shared)
        {
            
            spmv_csr(row_ptr_t, col_ind_t, values_t, num_rows, row_task, p_x, p_y);
            #pragma omp barrier
            #pragma omp for simd private(i) schedule(static)
            for(i = 0; i < num_rows; i++){
                p_x[i] = p_y[i]/1e2 + 1.0;
                p_y[i] = 0.0;
            }
        }
        
    }
    ela = (omp_get_wtime() - start)*1000;
    nthreads = omp_get_max_threads();
    printf("Finish at repeat:%d,use threads:%d, use time:%.4f ms\n",num_repeat,nthreads,ela);
    double err = 0;
    for(i = 0; i < num_rows; i++){
        err = max_function(err,fabs(p_x[i] - x[i]));
    }
    printf("L1_err:%.3e\n",err);
    

    return 0;
}

