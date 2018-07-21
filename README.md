# Sparse Tensor Note

## Purpose

This note tries to summarize the current state of sparse tensor in pytorch. It describes important invariance and properties of sparse tensor, and various things need to be fixed (e.g. empty sparse tensor). It also shows some details of sparse operators.


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

### More approaches in creating a sparse tensor
```
torch.sparse.FloatTensor()
torch.sparse.DoubleTensor()
torch.zeros(..., layout=torch.sparse_coo, ...)
```
Should we unify/reduce different approaches?


## Sparse representation

Currently, our sparse tensors are hybrid tensors, with a mix of sparse dims and dense dims. We keep track of `nnz`, `sparseDims`, `denseDims`, a `indices tensor of size = (sparseDims, nnz)`, and a `values tensor of size (nnz, size[sparseDims:])`. Additionally, we have a flag to note if the tensor is `coalesced`.

### Should we keep this representation?

- Our currently hybrid representation has some issues. It's difficult to support operators on hybrid tensors, *especially because only embedding uses hybrid*. Furthermore, some functions have ambiguous outputs on hybrid tensors (e.g. transposing a dense dim and a sparse dim is ambiguous). Because of this, it makes sense to have embedding has a special case, and only support “true” sparse tensors with denseDims = 0.

- Many sparse libraries use CSR because it's really efficient for ops such as mm. Ideally, we'd like to make use of these libraries, but it's difficult: CSR can't represent more than 2D, but COO requires a lot of conversions. Potential solution: caching CSR representations?

- Caffe2 uses per-row counts to represent sparse tensors, and they have some level of sparse support.

### A couple caveats with our current system

- Things we should always keep in mind
```
Ideal INVARIANTS:
  _sparseDims: range [0, len(shape)]; _sparseDims + _denseDims = len(shape)
  _denseDims : range [0, len(shape)]; _sparseDims + _denseDims = len(shape)
  _indices.shape: dimensionality: 2,  shape: (_sparseDims, nnz)
  _values.shape:  dimensionality: 1 + _denseDims.  shape: (nnz, shape[_sparseDims:])

Actual INVARIANT differences:
  1) _sparseDims: range [1, len(shape)] (i.e. we don't allow 0 sparse dimensions)
  2) when nnz = 0, there is strange behavior because we lack 0-dimensional sparse tensors.  Namely:
     dimensionality == 0, _sparseDims == 0, _denseDims == 0, _indices.shape == {0}, _values.shape == {0}
  3) For both _indices.shape and _values.shape, the nnz dimension may be larger than nnz
  4) For _values.shape, the non-nnz dimensions may be smaller than the corresponding dimension size, e.g.
     a shape (2,3) sparse tensor with _sparseDims == 1, may have _values.shape: (nnz, <=2, <=3).
```

- We're keeping track of nnz, sparseDims, denseDims and coalesced independently of the indices and values tensors. For instance, we may need to write the following code to maintain the invariance:
```
_get_sparse_impl(r)->set_indices_and_values(r_indices, r_values);
_get_sparse_impl(r)->set_nnz(t._nnz());
_get_sparse_impl(r)->set_coalesced(t.is_coalesced());
```

- Without support for 0-dim tensors, we cannot have proper support for scalar sparse tensors. This makes it unclear how to implement ops like reduction ops, which is currently represented as a dense scalar. Maybe there is nothing wrong with the current representation, we just need to be minded that sparse ops can return a scalar. 

- The nnz and the nnz dimension in two indices and value tensors may not be the same, some cuda kernels rely on this.

- In values, the non-nnz dimensions might not actually match the dimensions of the sparse tensor

- We have dim == sparseDims + denseDims, but sparseDims cannot be 0. On one hand, this makes sense because a sparse tensor should have at least one sparse dim, but at the same time, it's in conflict with the idea of a scalar sparse tensor.

- An empty sparse tensor has indices of dim = 1, which means we have to check `nnz != 0` everywhere in using a TensorAccessor of indices.

### Some of these issues are fixed by Will's PR (#9279)

Current behavoir of an empty sparse tensor:
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

>>> print('a._dimI = %d, a._sparseDims = %d, a._dimV = %d, a._denseDims = %d, a._nnz = %d' % 
          (a._dimI(), a._sparseDims(), a._dimV(), a._denseDims(), a._nnz()))
a._dimI = 0, a._sparseDims = 2, a._dimV = 0, a._denseDims = 0, a._nnz = 0
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

- An empty sparse tensor has indices of dim = 1, which means we have to check `nnz != 0` everywhere in using a TensorAccessor of indices.


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

A lot of operators on sparse tensors have densified gradients. e.g., `log1p` would make all 0 inputs have a gradient of 1, and it densifies the sparse tensor gradients. Currently we have to raise errors in the backward of these operators. Some potential ways to fix:

- Define special sparse operations so that they operate only on the nnz. This works well for functions that take a single input tensor. For example, we can define sparse log1p such that implicit 0s do not participate, and it would allow us to have a sparse gradient tensor. We can call this operator `s_log1p` or `nnz_log1p` (#1369 has similar discussions). However, we are not sure if they are really what users want, because these operators have different behaviors to their dense counterpart.

- Return a dense gradient. We can have backward functions able to handle both of dense and sparse gradients. No special treatment needed during autograd, just more functions to be written.

- Return the gradient as a sparse tensor. Even though we know a tensor might be completely dense, we can still choose to return its sparse form. This is awful for performance.

- No backward for densified gradients of sparse operator. Users need to call `to_dense` and apply dense operators instead. This will not work without backward for `to_dense`, which itself will have a densified gradients, and it brings us back to the 2nd proposal.


## Functions

### Pointwise one tensor math

|functions|dense grad|
|---|:---:|
|pow|N|
|log1p|Y|
|div_ / div(Sparse, Scalar)|Y|
|mul_ / mul(Sparse, Scalar)|Y|

All pointwise one tensor calls dense couterpart ops on `_values`, so maybe a macro can be written to cover them all.


### Pointwise two tensor math

|functions|formula|dense grad|
|---|---|:---:|
|add_ / add(Sparse, Sparse, Scalar) → Sparse|add(T, S, alpha) = T + alpha * S|Y|
|add_ / add(Dense, Sparse, Scalar) → Dense|add(T, S, alpha) = T + alpha * S|Y|
|sub_ / sub(Sparse, Sparse, Scalar) → Sparse|sub(T, S, alpha) = T - alpha * S|Y|
|mul_ / mul(Sparse, Sparse) → Sparse|mul(T, S) = T * S|Y|

All pointwise two tensor functions have properly optimized CUDA kernel except for `mul_ / mul`:
- `add_ / add(Sparse, Sparse, Scalar) → Sparse` returns cat(Sparse, Scalar * Sparse)
- `add_ / add(Dense, Sparse, Scalar) → Dense` parallelizes over nnz
- `sub_ / sub(Sparse, Sparse, Scalar) → Sparse` calls `add`

`mul_ / mul(Sparse, Sparse) → Sparse` needs parallelized CUDA kernels, possible directions:
1. write customized kernel to parallelized over nnz of 1st sparse tensor
2. use cuSPARSE?

### BLAS

|functions|formula|dense grad|
|---|---|:---:|
|addmm(Dense, Sparse, Dense, Scalar, Scalar) → Dense|addmm(T, S, D, beta, alpha) = beta * T + alpha * matmul(S, D)|Y|
|sspaddmm(Sparse, Sparse, Dense, Scalar, Scalar) → Sparse|sspaddmm(T, S, D, beta, alpha) = beta * T + alpha * matmul(S, D)|Y|
|mm(Sparse, Dense) → Dense|mm(S, D) = matmul(S, D)|Y|
|smm(Sparse, Dense) → Sparse|smm(S, D) = matmul(S, D)|Y|
|hspmm(Sparse, Dense) → HybridSparse|hspmm(S, D) = matmul(S, D)|Y|
|spmm(Sparse, Dense) → Dense|spmm(S, D) = matmul(S, D)|Y|

Functions with CUDA kernel optimized are `mm`, `addmm` and `hspmm`. `addmm` and `hspmm` use cuSPARSE (cusparseScsrmm2 and cusparseDcsrmm2) in CUDA kernel, and `mm` calls `addmm`. However, `smm` and `sspaddmm` don't have CUDA support (gets removed?).


### Others

|functions|dense grad|
|---|:---:|
|clone|NA|
|norm|N|
|zero_|N|
|t_ / t|N|


## Optimizers

- optim.SGD (CUDA and CPU)
- optim.SparseAdam (CUDA and CPU) - lazy version of Adam algorithm
- optim.Adagrad (CPU)


## Future work

### TODO functions

|Functions|dense grad|
|---|:---:|
|nonzero|           N|
|sum|               Y|
|copy_|             N|
|narrow|            N|
|select_index|      N|
|mul_ / mul(S, D)|  Y|
|cuda|              NA|
|F.linear|          Y|
|softmax|           N|
|cat|               N|
|max|               N|
|bmm(S, D)|         Y|

- There is a list of pointwise functions for sparse can be implemented by calling dense ops on their `_values`, some helpers or macros can be written to make all of these ops available for sparse.
```
abs, acos, asin, atan, ceil, cos, cosh, erf, exp, expm1, floor, 
log, log10, log2, round, sin, sinh, sqrt, rsqrt, tan, trunc
```

