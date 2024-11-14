import unittest
import sys
from HMM import HMM


class TestHMM(unittest.TestCase):
    def setUp(self):
        self.hmm = HMM()
        self.hmm.load('cat')

    def test_transitions_structure(self):
        # test correct transitions structure

        # test initial state probabilities
        self.assertIn('#', self.hmm.transitions)
        self.assertEqual(float(self.hmm.transitions['#']['happy']), 0.5)
        self.assertEqual(float(self.hmm.transitions['#']['grumpy']), 0.5)
        self.assertEqual(float(self.hmm.transitions['#']['hungry']), 0.0)

        # test transition probabilities
        self.assertEqual(float(self.hmm.transitions['happy']['grumpy']), 0.1)
        self.assertEqual(float(self.hmm.transitions['grumpy']['happy']), 0.6)
        self.assertEqual(float(self.hmm.transitions['hungry']['hungry']), 0.3)

    def test_emissions_structure(self):
        # test correct emissions structure

        # test emission probabilities
        self.assertEqual(float(self.hmm.emissions['happy']['silent']), 0.2)
        self.assertEqual(float(self.hmm.emissions['grumpy']['meow']), 0.4)
        self.assertEqual(float(self.hmm.emissions['hungry']['purr']), 0.2)

if __name__ == '__main__':
    unittest.main()
