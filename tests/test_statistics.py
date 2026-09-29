import unittest
import numpy as np
from breastseg.statistics import bootstrap_ci,holm,paired

class StatsContracts(unittest.TestCase):
    def test_holm_monotonic(self): np.testing.assert_allclose(holm([.04,.01,.03]),[.06,.03,.06])
    def test_identical_pairs(self):
        p=paired(np.arange(5),np.arange(5));self.assertEqual(p['p'],1);self.assertEqual(p['ci95'],[0,0])
    def test_bootstrap_reproducible(self): self.assertEqual(bootstrap_ci([1,2,3]),bootstrap_ci([1,2,3]))
