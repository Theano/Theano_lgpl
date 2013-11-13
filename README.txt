This repository contain Theano extension that reuse other people code
that is licensed under LGPL. To make sure there isn't license problem,
we keep them in a separate repository.

Currently, you just need to "import theano_lgpl" before compiling a
Theano function in your script to have the speed up provided by the
code here. All the rest is automatic.

Currently this provide c code for the elemwise operation chi2sf. It
work on CUDA.

To install for the current user:

    pip install --user git+git://github.com/Theano/Theano_lgpl.git

To install in the os:

    pip install git+git://github.com/Theano/Theano_lgpl.git
