import unittest
from transferlab.metrics import binomial, summarize


class MetricsTests(unittest.TestCase):
    def row(self, label, sc, tc, sa, ta):
        return dict(label=label, source_clean=sc, target_clean=tc,
                    source_adv=sa, target_adv=ta, l2=0., linf=0.)

    def test_clean_errors_are_not_attack_successes(self):
        rows = [self.row(0, 1, 1, 1, 1), self.row(0, 0, 0, 0, 0)]
        result = summarize(rows)
        self.assertEqual(result['adversarial_accuracy']['rate'], .5)
        self.assertEqual(result['pair_transfer']['rate'], 0)
        self.assertIsNone(result['conditional_transfer']['rate'])

    def test_pair_and_source_conditioning_are_distinct(self):
        rows = [self.row(0, 0, 0, 1, 1), self.row(0, 0, 0, 0, 1),
                self.row(0, 0, 0, 1, 0), self.row(0, 1, 0, 1, 1)]
        r = summarize(rows)
        self.assertEqual(r['pair_transfer']['total'], 3)
        self.assertAlmostEqual(r['pair_transfer']['rate'], 2/3)
        self.assertEqual(r['conditional_transfer']['rate'], .5)

    def test_empty_denominators_and_intervals(self):
        self.assertIsNone(binomial(0, 0)['rate'])
        self.assertLess(binomial(0, 10)['ci95'][1], .3)
        self.assertGreater(binomial(10, 10)['ci95'][0], .7)
        with self.assertRaises(ValueError):
            binomial(11, 10)
        with self.assertRaises(ValueError):
            summarize([])


if __name__ == '__main__':
    unittest.main()
