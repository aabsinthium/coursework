import pert as pt
import numpy as np
import pymc3 as pm
import theano.tensor as tt
from pymc3.distributions import draw_values, generate_samples
from pymc3.distributions.dist_math import betaln, bound


def _pert_clip_values(a, c):
    return {
        dtype: (np.nextafter(a, a + 1, dtype=dtype), np.nextafter(c, c - 1, dtype=dtype))
        for dtype in ["float16", "float32", "float64"]
    }


def clipped_pert_rvs(a, b, c, size=None, dtype="float64"):
    out = pt.PERT(a, b, c).rvs(size).astype(dtype)
    lower, upper = _pert_clip_values(a, c)[dtype]
    return np.maximum(np.minimum(out, upper), lower)


class pert(pm.distributions.continuous.BoundedContinuous):
    def __init__(self, a=None, b=None, c=None, *args, **kwargs):
        self.a = tt.as_tensor_variable(a)
        self.b = tt.as_tensor_variable(b)
        self.c = tt.as_tensor_variable(c)

        self.mean = (a + 6 * b + c) / 8
        self.median = (a + 4 * b + c) / 6

        super().__init__(lower=a, upper=b, *args, **kwargs)

    def logp(self, value):
        alpha = 1 + 4 * (self.b - self.a) / (self.c - self.a)
        beta = 1 + 4 * (self.c - self.b) / (self.c - self.a)
        ab = alpha + beta

        arg1 = value - self.a
        arg2 = self.c - value
        arg3 = self.c - self.a

        logarg1 = tt.log(arg1)
        logarg2 = tt.log(arg2)
        logarg3 = tt.log(arg3)

        logp = (
                tt.switch(tt.eq(alpha, 1), 0, (alpha - 1) * logarg1)
                + tt.switch(tt.eq(beta, 1), 0, (beta - 1) * logarg2)
                - tt.switch(tt.eq(ab, 1), 0, (ab - 1) * logarg3)
                - betaln(alpha, beta)
        )

        return bound(logp, value >= self.a, value <= self.c, alpha > 0, beta > 0)


    def random(self, point=None, size=None):
        a, b, c = draw_values([self.a, self.b, self.c], point=point, size=size)
        return generate_samples(clipped_pert_rvs, a, b, c, dist_shape=self.shape, size=size)
