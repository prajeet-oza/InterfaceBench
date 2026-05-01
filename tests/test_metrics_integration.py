import os
import unittest
import tempfile
import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.io import read
from interfacebench.metrics.metrics_utils import (
    compute_min_distance,
    compute_lindemann_index,
    compute_velocity_distribution_divergence
)

class TestMetricsIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.traj_path = os.path.join(self.temp_dir.name, "test.traj")
        
        # Build a miniature MD trajectory where atoms get closer over time
        t = Trajectory(self.traj_path, 'w')
        self.initial_min_dist = 2.0
        for i in range(10):
            # Gradually drift the atoms closer to simulate dynamics
            atoms = Atoms('Cu2', positions=[[0, 0, 0], [self.initial_min_dist - i*0.1, 0, 0]], cell=[5, 5, 5], pbc=True)
            atoms.set_momenta(np.array([[1.0 + i, 0, 0], [-1.0 - i, 0, 0]]))
            atoms.set_masses([63.546] * 2)
            t.write(atoms)
        t.close()
        
        self.trajectory = read(self.traj_path, index=':')

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_min_distance(self):
        """Test that min_distance correctly flags a violation."""
        # The threshold is 0.5 * initial_min_dist = 1.0.
        # The distance becomes 2.0 - 10*0.1 = 1.0 at frame 10, but we have 10 frames (0-9).
        # At frame 9, dist is 2.0 - 9*0.1 = 1.1. So it should be physical.
        res_physical = compute_min_distance(self.trajectory, use_half_initial=True)
        self.assertTrue(res_physical['is_physical'])

        # Now test a violation
        res_unphysical = compute_min_distance(self.trajectory, fixed_threshold=1.5)
        self.assertFalse(res_unphysical['is_physical'])
        self.assertEqual(res_unphysical['first_violation'], 5) # dist = 2.0 - 5*0.1 = 1.5
        
    def test_compute_lindemann_index(self):
        """Test that a simple trajectory returns a valid Lindemann index dict."""
        res = compute_lindemann_index(self.trajectory, local_threshold=0.2, global_threshold=0.15)
        self.assertFalse(res['is_physical'])
        self.assertIn('q_global', res)
        
    def test_compute_velocity_distribution_divergence(self):
        """Test that a simple trajectory can be parsed for MB stats."""
        res = compute_velocity_distribution_divergence(self.trajectory, temperature=300, threshold=0.2, start_at_frame=0)
        self.assertIn('is_physical', res)
        self.assertIn('hellinger_distances', res)

if __name__ == "__main__":
    unittest.main()