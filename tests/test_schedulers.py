import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from interfacebench.simulation.ase_scheduler import AseScheduler
from interfacebench.simulation.vasp_scheduler import VaspScheduler
from interfacebench.generators.db_manager import DatabaseManager
from interfacebench.generators.db_schema import InterfaceRecord, SlabRecord
from interfacebench.simulation.db_schema import SimulationRecord
from pymatgen.core import Structure, Lattice

class TestSchedulers(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_schedulers.db")
        self.work_dir = os.path.join(self.temp_dir.name, "test_work_dir")
        self.db_manager = DatabaseManager(self.db_path)
        
        # Insert real physical structure data into the database
        self.struct_dict = Structure(Lattice.cubic(3.0), ["Cu"], [[0,0,0]]).as_dict()
        with self.db_manager.SessionLocal() as session:
            film = SlabRecord(base_name="Cu", miller="100", termination="Cu", natoms=1, structure_dict=self.struct_dict)
            subs = SlabRecord(base_name="Ag", miller="100", termination="Ag", natoms=1, structure_dict=self.struct_dict)
            session.add_all([film, subs])
            session.commit()
            
            interface = InterfaceRecord(film_id=film.id, subs_id=subs.id, natoms=2, structure_dict=self.struct_dict)
            session.add(interface)
            session.commit()
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_ase_scheduler_init(self):
        """Test ASE Scheduler safely binds to DB and creates working dir"""
        scheduler = AseScheduler(self.db_path, self.work_dir)
        self.assertTrue(Path(self.work_dir).exists())
        self.assertTrue(os.path.exists(self.db_path))
        
    def test_vasp_scheduler_init(self):
        """Test VASP Scheduler safely binds to DB and creates working dir"""
        scheduler = VaspScheduler(self.db_path, self.work_dir)
        self.assertTrue(Path(self.work_dir).exists())

    def test_ase_scheduler_real_generation(self):
        """Test physical directory and script creation using real database queries"""
        scheduler = AseScheduler(self.db_path, self.work_dir)
        scheduler.generate_inputs(models=[{'tag': 'test_model', 'model_code': 'None', 'model_import': ''}], sim_type="relax")
        
        expected_dir = Path(self.work_dir) / "ase_relax_Cu_Ag_1_test_model"
        self.assertTrue(expected_dir.exists())
        self.assertTrue((expected_dir / "POSCAR").exists())
        self.assertTrue((expected_dir / "run_ase.py").exists())
        self.assertTrue((expected_dir / "kwargs_config.json").exists())

    @patch('interfacebench.simulation.vasp_scheduler.MPRelaxSet')
    def test_vasp_scheduler_real_db_generation(self, mock_relax_set):
        """Test VASP directory creation (Mocking MPRelaxSet to bypass POTCAR requirements)."""
        scheduler = VaspScheduler(self.db_path, self.work_dir)
        
        scheduler.generate_inputs(sim_type="relax")
        
        expected_dir = Path(self.work_dir) / "vasp_relax_Cu_Ag_1"
        self.assertTrue(expected_dir.exists())
        self.assertEqual(mock_relax_set.return_value.write_input.call_count, 3)
        mock_relax_set.return_value.write_input.assert_any_call(expected_dir)

if __name__ == "__main__":
    unittest.main()