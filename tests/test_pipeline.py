import os
import unittest
import tempfile
import pandas as pd
from pymatgen.core import Structure, Lattice
from interfacebench.pipeline import InterfaceBenchPipeline

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "structures.db")
        self.work_dir = os.path.join(self.temp_dir.name, "simulations")
        self.csv_path = os.path.join(self.temp_dir.name, "inputs.csv")
        
        # Change to temp directory so bulk CIFs are discoverable
        self.old_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)
        
        # Setup generic CIF files and inputs
        Structure(Lattice.cubic(3.0), ["Cu"], [[0,0,0]]).to(filename="Cu.cif")
        Structure(Lattice.cubic(3.1), ["Ag"], [[0,0,0]]).to(filename="Ag.cif")
        
        pd.DataFrame({
            'film': ['Cu'], 'subs': ['Ag'],
            'film_miller': ['1 0 0'], 'subs_miller': ['1 0 0'],
        }).to_csv(self.csv_path, index=False)
        
        self.pipeline = InterfaceBenchPipeline(db_path=self.db_path, work_dir=self.work_dir)
        
    def tearDown(self):
        os.chdir(self.old_cwd)
        self.temp_dir.cleanup()

    def test_full_pipeline_execution(self):
        # Generate physical inputs in the temporary SQLite DB
        self.pipeline.generate_structures(input_source=self.csv_path)
        self.assertTrue(os.path.exists(self.db_path))
        
        # Verify that scheduling works and properly reads the DB
        models = [{'tag': 'test_model', 'model_code': 'None', 'model_import': ''}]
        self.pipeline.simulate_mlip_tasks(models=models, sim_type="relax")
        self.assertTrue(os.path.exists(self.work_dir))

if __name__ == '__main__':
    unittest.main()