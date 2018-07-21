# Sparse Tensor Summary

## Purpose

## Sparse representation

## Semantics

## Functions

### Pointwise one tensor math

|functions|need autograd|dense grad|
|---|:---:|:---:|
|pow|Y|N|
|log1p|Y|Y|
|div_ / div(Sparse, Scalar)|Y|Y|
|mul_ / mul(Sparse, Scalar)|Y|Y|

All pointwise one tensor calls dense couterpart ops on `_values`. Specialized backward functions are needed for those with dense grad. The backward grad tensor will be a densified sparse tensor if we use the same formula in the backward of dense tensor. This is not ideal because it requires more memory and runs slower. Also for `pow`, we need to return a sparse grad in its sparse backward function.


### Pointwise two tensor math

|functions|need autograd|dense grad|
|---|:---:|:---:|
|add_ / add(Sparse, Sparse, Scalar) → Sparse|Y|Y|
|add_ / add(Dense, Sparse, Scalar) → Dense|Y|Y|
|sub_ / sub(Sparse, Sparse, Scalar) → Sparse|Y|Y|
|mul_ / mul(Sparse, Sparse) → Sparse|Y|Y|

All pointwise two tensor functions have properly optimized CUDA kernel except for `mul_ / mul` (its CUDA kernel needs to be parallelized):
- `add_ / add(Sparse, Sparse, Scalar) → Sparse` returns cat(Sparse, Scalar * Sparse)
- `add_ / add(Dense, Sparse, Scalar) → Dense` parallelizes over nnz
- `sub_ / sub(Sparse, Sparse, Scalar) → Sparse` calls `add`


### BLAS

|functions|need autograd|dense grad|
|---|:---:|:---:|
|addmm(Dense, Sparse, Dense, Scalar, Scalar) → Dense|Y|Y|
|sspaddmm(Sparse, Sparse, Dense, Scalar, Scalar) → Sparse|Y|Y|
|mm(Sparse, Dense) → Dense|Y|Y|
|smm(Sparse, Dense) → Sparse|Y|Y|
|hspmm(Sparse, Dense) → HybridSparse|Y|Y|
|spmm(Sparse, Dense) → Dense|Y|Y|


### Others

|functions|need autograd|dense grad|
|---|:---:|:---:|
|clone|N|N|
|norm|Y|N|
|zero_|N|N|
|t_ / t|Y|N|


## Optimizers

## Future work

### TODO functions

|Functions|need autograd|dense grad|
|---|:---:|:---:|
|nonzero|         Y|  N|
|sum|             Y|  Y|
|copy_|           Y|  N|
|narrow|          Y|  N|
|select_index|    Y|  N|
|mul_ / mul(S, D)|Y|  Y|
|cuda|            N|  N|
|F.linear|        Y|  Y|
|softmax|         Y|  N|
|cat|             Y|  N|
|max|             Y|  N|
|bmm(S, D)|       Y|  Y|
