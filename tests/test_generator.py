import os
import unittest
import tempfile
import pandas as pd
from pymatgen.core import Structure, Lattice
from interfacebench.generators.generator import GeneratorPipeline
from interfacebench.generators.db_schema import InterfaceRecord, SlabRecord
from interfacebench.simulation.db_schema import SimulationRecord

class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        
        # Create physical CIF files to run through the real generator
        s1 = Structure(Lattice.cubic(3.0), ["Cu"], [[0,0,0]])
        s2 = Structure(Lattice.cubic(3.1), ["Ag"], [[0,0,0]])
        s1.to(filename=os.path.join(self.temp_dir.name, "Cu.cif"))
        s2.to(filename=os.path.join(self.temp_dir.name, "Ag.cif"))
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_run_from_file_csv(self):
        csv_path = os.path.join(self.temp_dir.name, "inputs.csv")
        df = pd.DataFrame({
            'film': ['Cu'], 'subs': ['Ag'],
            'film_miller': ['1 0 0'], 'subs_miller': ['1 1 1']
        })
        df.to_csv(csv_path, index=False)
        
        pipeline = GeneratorPipeline(db_path=self.db_path, base_structure_loc=self.temp_dir.name)
        pipeline.run_from_file(csv_path)
        
        with pipeline.db_manager.SessionLocal() as session:
            interfaces = session.query(InterfaceRecord).all()
            slabs = session.query(SlabRecord).all()
            self.assertTrue(len(interfaces) > 0, "No interfaces were saved to the real DB.")
            self.assertTrue(len(slabs) > 0, "No slabs were saved to the real DB.")

    def test_run_from_file_yaml(self):
        yaml_path = os.path.join(self.temp_dir.name, "inputs.yaml")
        with open(yaml_path, 'w') as f:
            f.write("- film: Cu\n  subs: Ag\n  film_miller: [1, 0, 0]\n  subs_miller: [1, 1, 1]\n")
            
        pipeline = GeneratorPipeline(db_path=self.db_path, base_structure_loc=self.temp_dir.name)
        pipeline.run_from_file(yaml_path)

        with pipeline.db_manager.SessionLocal() as session:
            self.assertTrue(len(session.query(InterfaceRecord).all()) > 0)
    
if __name__ == "__main__":
    unittest.main()