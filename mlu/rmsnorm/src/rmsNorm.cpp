#include "cnnl.h"
#include "cnrt.h"
#include "cnnl_extra.h"
#include <vector>
#include <sys/time.h>

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
void rmsnormCnnlDevice(float *source, float *destination, float *weight, int nDim, int *shape, float eps, cnnlHandle_t &handle, cnrtQueue_t &queue)
{
    cnnlTensorDescriptor_t yDesc, xDesc, wDesc;
    cnnlCreateTensorDescriptor(&yDesc);
    cnnlCreateTensorDescriptor(&xDesc);
    cnnlCreateTensorDescriptor(&wDesc);

    std::vector<int> inDim(nDim);
    std::vector<int> outDim(nDim);
    std::vector<int> weightDim(1);
    for (uint64_t i = 0; i < nDim; i++) {
        inDim[i] = shape[i];
        outDim[i] = shape[i];
    }
    weightDim[0] = shape[nDim - 1];

    cnnlSetTensorDescriptor(
        xDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
        inDim.size(), inDim.data());
    cnnlSetTensorDescriptor(
        yDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
        outDim.size(), outDim.data());
    cnnlSetTensorDescriptor(
        wDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
        weightDim.size(), weightDim.data());
    
    cnnlFuseNormDescriptor_t opDesc;
    cnnlCreateFuseNormDescriptor(&opDesc);
    cnnlSetFuseNormDescriptor(opDesc, eps, 1.0, true,
                              false, false, false, false,
                              CNNL_DTYPE_FLOAT, CNNL_TRANSFORMER_RMSNORM);

    size_t wsSize;
    cnnlGetFuseNormWorkspaceSize(handle, opDesc, xDesc, &wsSize);

    void *workspace;
    cnrtMalloc(&workspace, wsSize);

    cnrtNotifier_t start = nullptr, end = nullptr;
    CNRT_CHECK(cnrtNotifierCreate(&start));
    CNRT_CHECK(cnrtNotifierCreate(&end));
    CNRT_CHECK(cnrtPlaceNotifier(start, queue));

    cnnlFuseNorm(handle, opDesc, xDesc, source,
                 wDesc, weight, nullptr, nullptr,
                 nullptr, nullptr, nullptr, nullptr,
                 workspace, wsSize, yDesc, destination, nullptr, nullptr);

    CNRT_CHECK(cnrtPlaceNotifier(end, queue));
    CNRT_CHECK(cnrtQueueSync(queue));

    float timeTotal;
    CNRT_CHECK(cnrtNotifierDuration(start, end, &timeTotal));
    printf("cnnl rmsnorm queue time:%.3f ms\n", timeTotal / 1000.0);

    cnrtFree(workspace);
    cnnlDestroyFuseNormDescriptor(opDesc);

    cnnlDestroyTensorDescriptor(xDesc);
    cnnlDestroyTensorDescriptor(yDesc);
    cnnlDestroyTensorDescriptor(wDesc);
    CNRT_CHECK(cnrtNotifierDestroy(start));
    CNRT_CHECK(cnrtNotifierDestroy(end));
}
void rmsnormCnnl(float *host_destination, float *host_src, float *host_weight, int nDim, int *shape, float eps)
{
    int num = 1;
    for (int s = 0; s < nDim; s++)
    {
        num *= shape[s];
    }
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。
    float *mlu_destination;
    float *mlu_src;
    float *mlu_weight;

    CNRT_CHECK(cnrtMalloc((void **)&mlu_destination, num * sizeof(float)));
    CNRT_CHECK(cnrtMalloc((void **)&mlu_src, num * sizeof(float)));
    CNRT_CHECK(cnrtMalloc((void **)&mlu_weight, shape[nDim - 1] * sizeof(float)));

    CNRT_CHECK(cnrtMemcpy(mlu_src, host_src, num * sizeof(float), cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtMemcpy(mlu_weight, host_weight, shape[nDim -1] * sizeof(float), cnrtMemcpyHostToDev));

    //----------------------------
    double st, ela;
    st = get_walltime();

    rmsnormCnnlDevice(mlu_src, mlu_destination, mlu_weight, nDim, shape, eps, handle, queue);

    ela = get_walltime() - st;

    CNRT_CHECK(cnrtMemcpy(host_destination, mlu_destination, num * sizeof(float), cnrtMemcpyDevToHost));

    printf("cnnl rmsnorm Total Time: %.3f ms\n", ela * 1000);
    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));

    cnrtFree(mlu_destination);
    cnrtFree(mlu_src);
    cnrtFree(mlu_weight);
}




