import unittest
from unittest.mock import patch, MagicMock
from interfacebench.metrics.metrics_registry import energy_metric, min_distance, rms_distance_metric
from pymatgen.core import Structure

class TestMetricsRegistry(unittest.TestCase):
    
    def test_energy_metric_missing_files_graceful_fail(self):
        """Ensure the metric wrapper returns an error dict instead of crashing on missing paths"""
        res = energy_metric("fake_baseline_dir", "fake_test_dir")
        self.assertIn("error", res)
        self.assertEqual(res["error"], "Missing energy files")

    @patch('interfacebench.metrics.metrics_registry.load_final_structure')
    @patch('interfacebench.metrics.metrics_registry.load_final_energy')
    @patch('interfacebench.metrics.metrics_registry.compute_multipliers')
    def test_energy_metric_calculation_success(self, mock_mult, mock_load_energy, mock_load_struct):
        """Mocks out the files to test the internal dictionary generation and multiplier logic"""
        mock_mult.return_value = (1.0, 1.0)
        mock_load_struct.return_value = MagicMock()
        # Energies for: dft_int, ase_int, dft_film, dft_subs, ase_film, ase_subs
        mock_load_energy.side_effect = [-10.0, -9.5, -3.0, -4.0, -2.8, -3.9] 
        
        mock_interface = MagicMock()
        mock_interface.film.structure_dict = MagicMock(spec=Structure)
        mock_interface.subs.structure_dict = MagicMock(spec=Structure)
        
        kwargs = {
            'interface': mock_interface,
            'dft_film_dir': 'mock_dir', 'dft_subs_dir': 'mock_dir',
            'ase_film_dir': 'mock_dir', 'ase_subs_dir': 'mock_dir'
        }
        
        res = energy_metric("base_dir", "test_dir", use_bulk_ref=False, **kwargs)
        
        self.assertNotIn("error", res)
        self.assertIn("dft_interfacial_energy", res)
        self.assertIn("ase_interfacial_energy", res)
        
        # -10.0 - (1.0*-3.0) - (1.0*-4.0) = -3.0
        self.assertAlmostEqual(res["dft_interfacial_energy"], -3.0)
        
    @patch('interfacebench.metrics.metrics_registry.load_entire_trajectory')
    def test_min_distance_missing_files_graceful_fail(self, mock_load_traj):
        mock_load_traj.return_value = None
        res = min_distance("fake_baseline_dir", "fake_test_dir")
        self.assertIn("error", res)
        self.assertEqual(res["error"], "Missing trajectory files")
        
if __name__ == "__main__":
    unittest.main()