
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include "cuda_jacobi.h"
#define max_function(a, b) ((a) > (b) ? (a) : (b))

#define max_iter 20000

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
void read_non(const char *filename, int *num)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }

    fscanf(file, "%d %d %d\n", &num[0], &num[1], &num[2]); // 只能读取指针
    fclose(file);
}

void read_matrix(const char *filename, int *num, int *row_ptr_t, int *col_ind_t, double *values_t)
{
    int num_rows, num_vals;
    int i;
    num_rows = num[0];
    num_vals = num[2];
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }

    // 统计每一行非零元个数--------------------------------------------
    int row_occurances[num_rows];

    for (i = 0; i < num_rows; i++)
    {
        row_occurances[i] = 0; // init data
    }
    fscanf(file, "%d %d %d\n", &num[0], &num[1], &num[2]);
    int row, column;
    double value;
    while (fscanf(file, "%d %d %lf\n", &row, &column, &value) != EOF)
    {
        // Subtract 1 from row and column indices to match C format
        row--; // 非零元的行索引，必须要减一
        column--;

        row_occurances[row]++; // 此时第row个元素存储的是该row的非零元个数
    }

    // 使用CSR格式读取矩阵元素---------------------------------
    //  Set row offset
    int index = 0;
    for (i = 0; i < num_rows; i++)
    {
        row_ptr_t[i] = index; // 这个才是那个row offset数组
        index += row_occurances[i];
    } // row_ptr_t[i + 1] - row_ptr_t[i]表示第i行元素个数
    row_ptr_t[num_rows] = num_vals; // 最后一个元素就是非零元总数

    // 根据row offset 读取列索引
    //  Set the file position to the beginning of the file
    rewind(file);
    for (i = 0; i < num_vals; i++)
    {
        col_ind_t[i] = -1; // init data
    }
    fscanf(file, "%d %d %d\n", &num[0], &num[1], &num[2]); // 不可删除
    i = 0;
    while (fscanf(file, "%d %d %lf\n", &row, &column, &value) != EOF)
    {
        row--;
        column--;

        // Find the correct index (i + row_ptr_t[row]) using both row information and an index i
        while (col_ind_t[i + row_ptr_t[row]] != -1)
        { // col_ind_t[row_ptr_t[row]:row_ptr_t[row + 1]]表示第row非零元的列索引
            i++;
        } // i + row_ptr_t[row]表示第row行一共有row_ptr_t[row]个非零元，
        col_ind_t[i + row_ptr_t[row]] = column;
        values_t[i + row_ptr_t[row]] = value;
        i = 0;
    }

    fclose(file);
}

void read_rigacc(const char *file_rig, const char *file_acc, int *num, double *rig, double *u)
{
    int num_rows;
    int i;
    num_rows = num[0];
    int M, N;
    FILE *fptr_rig = fopen(file_rig, "r");
    if (fptr_rig == NULL)
    {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }
    fscanf(fptr_rig, "%d %d %d\n", &M, &N, &num_rows); // 不可删除
    double value;
    i = 0;
    while (fscanf(fptr_rig, "%lf\n", &value) != EOF)
    {
        // Subtract 1 from row and column indices to match C format
        rig[i] = value;
        i += 1;
    }
    fclose(fptr_rig);
    //-------------------------
    FILE *fptr_acc = fopen(file_acc, "r");
    if (fptr_acc == NULL)
    {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }
    fscanf(fptr_acc, "%d %d %d\n", &M, &N, &num_rows); // 不可删除
    i = 0;
    while (fscanf(fptr_acc, "%lf\n", &value) != EOF)
    {
        // Subtract 1 from row and column indices to match C format
        u[i] = value;
        i += 1;
    }
    fclose(fptr_acc);
}
void init0(double *u, int num_rows)
{ // init all = 0
    for (int i = 0; i < num_rows; i++)
    {
        u[i] = 0;
    }
}
void spmv_csr(const int *row_ptr, const int *col_ind, const double *values, const int num_rows, const double *x, double *y)
{
    int i;

    for (i = 0; i < num_rows; i++)
    {
        double sum = 0;
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];

        for (int j = row_start; j < row_end; j++)
        {
            sum += values[j] * x[col_ind[j]];
        }

        y[i] = sum;
    }
}
void Jacobi(const int *row_ptr, const int *col_ind, const double *values, const int num_rows, double *u_new, const double *rig)
{
    double u_old[num_rows];
    int k = 0, i;
    double tmp;
    while (k < max_iter)
    {
        for (i = 0; i < num_rows; i++)
        {
            u_old[i] = u_new[i];
        }
        for (i = 0; i < num_rows; i++)
        {
            double sum = 0;
            const int row_start = row_ptr[i];
            const int row_end = row_ptr[i + 1];

            for (int j = row_start; j < row_end; j++)
            {
                sum += values[j] * u_old[col_ind[j]];
                if (col_ind[j] == i)
                {
                    tmp = values[j];
                }
            }
            u_new[i] = u_old[i] + (rig[i] - sum) / tmp;
        }
        k += 1;
    }
}

double L1_err(double *u_new, double *u, int num_rows)
{
    int i, j;
    double err = 0;
    for (i = 0; i < num_rows; i++)
    {
        err = max_function(err, fabs(u_new[i] - u[i]));
    }

    return err;
}
int main(int argc, char **argv)
{
    const char *file_mat = "./matrix.mtx";
    const char *file_rig = "./rig.vec";
    const char *file_acc = "./uacc.vec";
    int num[3];
    read_non(file_mat, num);
    int num_rows, num_vals;
    num_rows = num[0];
    num_vals = num[2];
    printf("run jacobi, the length of vector = M * N:%d\n", num_rows);
    int row_ptr_t[num_rows + 1];
    int col_ind_t[num_vals];
    double values_t[num_vals];
    read_matrix(file_mat, num, row_ptr_t, col_ind_t, values_t);

    double u[num_rows], rig[num_rows];
    read_rigacc(file_rig, file_acc, num, rig, u);

    double err = 0;
    double jacobi_st, jacobi_ela;
    //---------------------
    double u_new[num_rows];

    //-------------------------------------
    init0(u_new, num_rows);
    jacobi_st = get_walltime();
    Jacobi(row_ptr_t, col_ind_t, values_t, num_rows, u_new, rig);
    jacobi_ela = get_walltime() - jacobi_st;
    err = L1_err(u_new, u, num_rows);
    printf("CPU jacobi use time:%.2f, L1_err::%.4e\n", jacobi_ela, err);
    double tmp[num_rows];
    for (int i = 0; i < num_rows; i++)
    {
        tmp[i] = u_new[i];
    }
    //----------------------------------------
    init0(u_new, num_rows);
    jacobi_st = get_walltime();
    cuda_solve(row_ptr_t, col_ind_t, values_t, num_rows, num_vals, u_new, rig);
    jacobi_ela = get_walltime() - jacobi_st;
    err = L1_err(u_new, tmp, num_rows);
    printf("GPU jacobi use time:%.2f, L1_err::%.4e\n", jacobi_ela, err);
    return 0;
}
