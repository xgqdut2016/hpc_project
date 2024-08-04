#include "cnnl.h"
#include "cnrt.h"
#include <vector>
#include <sys/time.h>

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
void softmaxCnnlDevice(float *source, float *destination, int nDim, int axis, int *shape, cnnlHandle_t &handle, cnrtQueue_t &queue)
{
    cnnlSoftmaxMode_t mode;
    std::vector<int> inDim = {1, 1, 1};
    std::vector<int> outDim = inDim;

    if (nDim >= 3)
    {
        if (axis == 0)
        {
            mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
            inDim[0] = shape[0];
            inDim[1] = shape[1];
            for (int i = 2; i < nDim; ++i)
            {
                inDim[2] *= shape[i];
            }
            outDim = inDim;
        }
        else if (axis == nDim - 1)
        {
            mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
            inDim[0] = shape[0];
            for (int i = 1; i < axis; ++i)
            {
                inDim[1] *= shape[i];
            }
            inDim[2] = shape[axis];
            outDim = inDim;
        }
        else
        {
            mode = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
            for (int i = 0; i < axis; ++i)
            {
                inDim[0] *= shape[i];
            }
            inDim[1] = shape[axis];
            for (int i = axis + 1; i < nDim; ++i)
            {
                inDim[2] *= shape[i];
            }
            outDim = inDim;
        }
    }
    else if (nDim == 2)
    {
        if (axis == 0)
        {
            mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
            inDim[0] = shape[0];
            inDim[1] = shape[1];

            outDim = inDim;
        }
        else
        {
            mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
            inDim[1] = shape[0];
            inDim[2] = shape[1];

            outDim = inDim;
        }
    }
    else
    {
        mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
        inDim[0] = shape[0];

        outDim = inDim;
    }
    cnnlTensorDescriptor_t aDesc, cDesc;
    cnnlCreateTensorDescriptor(&aDesc);
    cnnlSetTensorDescriptor(
        aDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
        inDim.size(), inDim.data());
    cnnlCreateTensorDescriptor(&cDesc);
    cnnlSetTensorDescriptor(
        cDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
        outDim.size(), outDim.data());
    float alpha = 1.0;
    float beta = 0.0;

    cnrtNotifier_t start = nullptr, end = nullptr;
    CNRT_CHECK(cnrtNotifierCreate(&start));
    CNRT_CHECK(cnrtNotifierCreate(&end));
    CNRT_CHECK(cnrtPlaceNotifier(start, queue));

    cnnlStatus_t stat =
        cnnlSoftmaxForward_v2(handle, CNNL_SOFTMAX_ACCURATE,
                              mode, CNNL_COMPUTATION_ULTRAHIGH_PRECISION,
                              &alpha, aDesc, source, &beta, cDesc, destination);
    CNRT_CHECK(cnrtPlaceNotifier(end, queue));
    CNRT_CHECK(cnrtQueueSync(queue));

    float timeTotal;
    CNRT_CHECK(cnrtNotifierDuration(start, end, &timeTotal));
    printf("cnnl softmax queue time:%.3f ms\n", timeTotal / 1000.0);
    if (stat != CNNL_STATUS_SUCCESS)
        return;
    cnnlDestroyTensorDescriptor(aDesc);
    cnnlDestroyTensorDescriptor(cDesc);
    CNRT_CHECK(cnrtNotifierDestroy(start));
    CNRT_CHECK(cnrtNotifierDestroy(end));
}
void softmaxCnnl(float *host_destination, float *host_src, int nDim, int axis, int *shape)
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

    CNRT_CHECK(cnrtMalloc((void **)&mlu_destination, num * sizeof(float)));
    CNRT_CHECK(cnrtMalloc((void **)&mlu_src, num * sizeof(float)));

    CNRT_CHECK(cnrtMemcpy(mlu_src, host_src, num * sizeof(float), cnrtMemcpyHostToDev));

    //----------------------------
    double st, ela;
    st = get_walltime();

    softmaxCnnlDevice(mlu_src, mlu_destination, nDim, axis, shape, handle, queue);

    ela = get_walltime() - st;

    CNRT_CHECK(cnrtMemcpy(host_destination, mlu_destination, num * sizeof(float), cnrtMemcpyDevToHost));

    printf("cnnl softmax Total Time: %.3f ms\n", ela * 1000);
    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));

    cnrtFree(mlu_destination);
    cnrtFree(mlu_src);
}



