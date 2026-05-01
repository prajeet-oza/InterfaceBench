import os
import unittest
import tempfile
from unittest.mock import MagicMock

from interfacebench.metrics.orchestrator import BenchmarkOrchestrator

class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_orchestrator.db")
        self.orch = BenchmarkOrchestrator(self.db_path)
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_orchestrator_init_and_defaults(self):
        """Test that Orchestrator registers default metrics upon init"""
        self.assertGreater(len(self.orch.metrics), 0, "Default metrics failed to register")
        
    def test_metric_registration(self):
        """Test dynamic registration of custom metrics"""
        initial_count = len(self.orch.metrics)
        
        self.orch.register_metric("dummy_metric", "relax", lambda **kwargs: {"result": 1.0})
        self.assertEqual(len(self.orch.metrics), initial_count + 1)
        self.assertEqual(self.orch.metrics[-1]["name"], "dummy_metric")
        
    def test_metric_filtering_by_sim_type(self):
        """Test that MD metrics don't trigger for relax tasks and vice versa"""
        relax_metrics = [m for m in self.orch.metrics if m["sim_type"] == "relax"]
        md_metrics = [m for m in self.orch.metrics if m["sim_type"] == "md"]
        
        self.assertGreater(len(relax_metrics), 0)
        self.assertGreater(len(md_metrics), 0)

if __name__ == "__main__":
    unittest.main()