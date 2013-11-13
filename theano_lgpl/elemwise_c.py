import theano
from theano import gof, tensor
from theano.scalar.basic import (BinaryScalarOp,
                                 upgrade_to_float,
                                 float_types)
from theano.scalar.basic_scipy import chi2sf
from theano.tensor.opt import register_specialize
imported_scipy_special = False
try:
    import scipy.special
    import scipy.stats
    imported_scipy_special = True
# Importing scipy.special may raise ValueError.
# See http://projects.scipy.org/scipy/ticket/1739
except (ImportError, ValueError):
    pass

import scalar_c

cchi2sf = tensor.elemwise.Elemwise(scalar_c.cchi2sf, name="cchi2sf")


@register_specialize
@gof.local_optimizer([tensor.Elemwise])
def local_chi2sf_to_cchi2sf(node):
    """This replace Elemwise{Chi2SF} -> Elemwise{CChi2SF}

    The new version have c code, so it can run on the GPU and have
    less overhead on the CPU and can be fused with other Elemwise.

    """
#    import pdb;pdb.set_trace()
    if node.op == tensor.Elemwise and node.op.scalar_op == chi2sf:
        return cchi2sf(*node.inputs)
