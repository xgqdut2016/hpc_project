# MLU注意事项
1：寒武纪kernel后缀是.mlu，比如说he.mlu 

2：编译命令为cncc he.mlu -o he --bang-mlu-arch=mtp_372 -O3 -lm 

## 1Dsoftmax
一维向量的softmax代码

## highsoftmax
高维向量的softmax代码

## random_sample
topk, topp,temperature采样的代码

## reduce_sum
sum规约代码

## rms_norm
rms norm算子代码

## rotary_embedding
rotary embedding算子代码

## softmax
softmax和库函数对比代码
