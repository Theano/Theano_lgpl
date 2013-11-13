from theano.tensor.tests.test_basic import (
    expected_chi2sf, _good_broadcast_unary_chi2sf,
    makeBroadcastTester, mode_no_scipy, skip_scipy)

from theano_lgpl.elemwise_c import cchi2sf

CChi2SFTester = makeBroadcastTester(
    op=cchi2sf,
    expected=expected_chi2sf,
    good=_good_broadcast_unary_chi2sf,
    eps=2e-10,
    mode=mode_no_scipy,
    skip=skip_scipy,
    name='CChi2SFTester')
