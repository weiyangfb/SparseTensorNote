# Sparse Tensor Note

## Purpose

This note tries to summarize the current state of sparse tensor in pytorch. It describes important invariants and properties of sparse tensor, various things that need to be fixed (e.g. empty sparse tensor), and shows some details of sparse operators.


## Semantics

### Construct a sparse tensor
```
# create a float sparse tensor in CPU
>>> indices = torch.LongTensor([[0, 0, 1], [0, 1, 1]])
>>> values = torch.FloatTensor([2, 3, 4])
>>> sizes = [2, 2]
>>> torch.sparse_coo_tensor(indices, values, sizes)
torch.sparse.FloatTensor of size (2,2) with indices:
tensor([[0, 0, 1],
        [0, 1, 1]])
and values:
tensor([2., 3., 4.])

# to_dense
>>> torch.sparse_coo_tensor(indices, values, sizes).to_dense()
tensor([[2., 3.],
        [0., 4.]])

# in CUDA
>>> torch.sparse_coo_tensor(indices, values, sizes, device=torch.device('cuda'))
torch.cuda.sparse.FloatTensor of size (2,2) with indices:
tensor([[0, 0, 1],
        [0, 1, 1]], device='cuda:0')
and values:
tensor([2., 3., 4.], device='cuda:0')
```


## Sparse representation

Currently, our sparse tensors are actually hybrid tensors, with some sparse dims and some dense dims. We keep track of nnz, sparseDims, denseDims, a size (sparseDims, nnz) indices tensor, and a size (nnz, ...) values tensor, where ... is the size of an individual value. Additionally, we have a flag to note if the tensor is coalesced.

### Should we keep this representation?

- Our currently hybrid representation has some issues. It's difficult to support operators on hybrid tensors, *especially because only embedding uses hybrid*. Furthermore, some functions have ambiguous outputs on hybrid tensors (e.g. transposing a dense dim and a sparse dim is ambiguous). Because of this, it makes sense to have embedding has a special case, and only support “true” sparse tensors.

- Many sparse libraries use CSR because it's really efficient for ops such as mm. Ideally, we'd like to make use of these libraries, but it's difficult: CSR can't represent more than 2D, but COO requires a lot of conversions. Potential solution: caching CSR representations?

- Caffe2 uses per-row counts to represent sparse tensors, and they have some level of sparse support.

### A couple caveats with our current system
- We're keeping track of `nnz`, `sparseDims`, `denseDims` and `is_coalesce` independently of the indices and values tensors

- Without support for 0-dim tensors, we cannot have proper support for scalar sparse tensors. This makes it unclear how to implement ops like reduction ops.

- The nnz and the nnz dimension in two indices and value tensors may not be the same, some cuda kernels rely on this

- In values, the non-nnz dimensions might not actually match the dimensions of the sparse tensor

- We have dim == sparseDims + denseDims, but sparseDims cannot be 0. On one hand, this makes sense because a sparse tensor should have at least one sparse dim, but at the same time, it's in conflict with the idea of a scalar sparse tensor.

### Some of these issues are fixed by Will's PR (#9279)

Current behavoir of empty sparse tensor:
```
>>> a = torch.sparse_coo_tensor([], [], [2, 3])
>>> i = a._indices()
>>> v = a._values()

>>> print(a)
torch.sparse.FloatTensor of size (2,3) with indices:
tensor([], dtype=torch.int64)
and values:
tensor([])

>>> print('i = %s' % i)
i = tensor([], dtype=torch.int64)

>>> print('v = %s' % v)
v = tensor([])

>>> print('a.dim = %d, i.dim = %d, v.dim = %d' % (a.dim(), i.dim(), v.dim()))
a.dim = 2, i.dim = 1, v.dim = 1

>>> print('a._dimI = %d, a._dimV = %d, a._nnz = %d' % (a._dimI(), a._dimV(), a._nnz()))
a._dimI = 0, a._dimV = 0, a._nnz = 0
```

When properly supported:
```
>>> import torch
>>> a=torch.sparse.DoubleTensor()
>>> a
torch.sparse.DoubleTensor of size (0,) with indices:
tensor([], size=(1, 0), dtype=torch.int64)
and values:
tensor([], dtype=torch.float64)

# empty sparse tensor to be a 1-dimensional tensor of size [0], 
# with sparseDims == 1 and denseDims == 0

# invariants:
#   _sparseDims + _denseDims = len(shape)
#   _indices.dim = 2, shape = (_sparseDims, nnz)
#   _values.dim = 1 + _denseDims, shape = (nnz, shape[_sparseDims:])
```

This fixes:

- the sparse tensor constructor accepts a scalar tensor (dim == 0) for the values tensor. This is kind of weird because you always need an nnz-dimension. However, the old code supported this and just expanded it into a dim=1 size=0 tensor. Currently we convert the scalar tensor to a dim=1 tensor.

- In ATen, sparse tensor has a different internal representation invariant: an "empty" sparse tensor has dimI == 1 and dimV == 0 (this is different from dimI == 0 and dimV == 0 we had before); this ensures that we maintain the invariant that dim == dimI + dimV. "Scalar" sparse tensors are made illegal

- when using TensorAccessor to access sparse indices, we need to check that nnz != 0 to make sure indices is 2D. This will go away when we properly support zero-size dimensions. 


### Autograd support
```
>>> a = torch.sparse_coo_tensor(indices, values, sizes, requires_grad=True)
>>> b = a * 2
>>> b.backward(torch.sparse_coo_tensor(indices, values, sizes))
>>> print(a)

torch.sparse.FloatTensor of size (2,2) with indices:
tensor([[0, 0, 1],
        [0, 1, 1]])
and values:
tensor([4., 6., 8.])
```

A lot of operators on sparse tensors don't have meaningful gradients for the non-nnz elements. e.g., log1p() would make all 0 inputs have a gradient of 1. Some potential ways to fix:

- Define sparse operations so that they operate only on the nnz. This works well for functions that take a single input tensor. For example, we can define sparse log1p such that implicit 0s do not participate, and it would allow us to have a sparse gradient tensor. 

- Return a dense gradient, and then run equivalent dense backwards functions for each previous sparse operation. Obviously this won't work well in conjunction with the first solution, because sparse ops and dense ops wouldn't be equivalent operations.

- Just return the gradient as a sparse tensor, even though we know it is completely dense. This is awful for performance.


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

All pointwise two tensor functions have properly optimized CUDA kernel except for `mul_ / mul`:
- `add_ / add(Sparse, Sparse, Scalar) → Sparse` returns cat(Sparse, Scalar * Sparse)
- `add_ / add(Dense, Sparse, Scalar) → Dense` parallelizes over nnz
- `sub_ / sub(Sparse, Sparse, Scalar) → Sparse` calls `add`

`mul_ / mul(Sparse, Sparse) → Sparse` needs parallelized CUDA kernels, possible directions:
1. write customized kernel to parallelized over nnz of 1st sparse tensor
2. use cuSPARSE?

### BLAS

|functions|need autograd|dense grad|
|---|:---:|:---:|
|addmm(Dense, Sparse, Dense, Scalar, Scalar) → Dense|Y|Y|
|sspaddmm(Sparse, Sparse, Dense, Scalar, Scalar) → Sparse|Y|Y|
|mm(Sparse, Dense) → Dense|Y|Y|
|smm(Sparse, Dense) → Sparse|Y|Y|
|hspmm(Sparse, Dense) → HybridSparse|Y|Y|
|spmm(Sparse, Dense) → Dense|Y|Y|

Functions with CUDA kernel well optimized are `mm`, `addmm` and `hspmm`. `addmm` and `hspmm` use cuSPARSE (cusparseScsrmm2 and cusparseDcsrmm2) in CUDA kernel, and `mm` calls `addmm`. `smm` and `sspaddmm` don't have CUDA support yet.


### Others

|functions|need autograd|dense grad|
|---|:---:|:---:|
|clone|N|N|
|norm|Y|N|
|zero_|N|N|
|t_ / t|Y|N|


## Optimizers

- optim.SGD (CUDA and CPU)
- optim.SparseAdam (CUDA and CPU) - lazy version of Adam algorithm
- optim.Adagrad (CPU)


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

All pointwise functions for sparse can potentially call dense ops on their `_values`, we can write some helpers or macros to make all of these ops available for sparse.
- abs, acos, asin, atan, ceil, cos, cosh, erf, exp, expm1, floor, log, log10, log2, round, sin, sinh, sqrt, rsqrt, tan, trunc


