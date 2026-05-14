# DGEMM

## RUN dgemm test on a single node.

### BUILDING

Use cmake and set the following arguments on the command line:

``` bash
-DBLAS_ROOT=<Root dir of installed blas package>
-DBLAS_NAME=<Name of blas package>
```

Supported blas names are shown in the `LIBRARIES SUPORTED` section below.

If necessary, point directly at the library and include dirs for the blas package with
`BLAS_LIB_DIR` and `BLAS_INCLUDE_DIR`.

You could compile the source file directly on the command line:

``` bash
# BLAS_NAME must be upper case.
$CC mt-dgemm.c -o mtdgemm -fopenmp -I${BLAS_INCLUDE_DIR} -L${BLAS_LIB_DIR} -l${BLAS_LIB} -DUSE_${BLAS_NAME}
```

### RUNNING

### LIBRARIES SUPPORTED

* cblas
* cublas
* cublasxt
* essl
* libsci
* libsci_acc
* mkl
* nvpl

The user can also choose `raw` which is a handcoded matrix-matrix multiply in C.

### CREDIT/CONTRIBUTING

Many thanks to the SNL team, Anthony, Doug, etc. Who initiated this project.