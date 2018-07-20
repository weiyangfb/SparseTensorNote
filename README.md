# Sparse Tensor Summary

## Purpose

## Sparse representation

## Semantics

## Functions

## Optimizers

## Future work

- functions

|Functions|need autograd|dense grad|
|---|:---:|:---:|
|nonzero|Y|N|
|sum|N|Y|
|copy_|Y|N|
|narrow|Y||
|select_index|Y||
|mul_ / mul(S, D)|Y|Y|
|cuda|N|N|
|F.linear|Y|Y|
|softmax|Y|N|
