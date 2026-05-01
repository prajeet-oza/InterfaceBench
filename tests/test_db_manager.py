import os
import unittest
import tempfile
from unittest.mock import MagicMock, patch
from interfacebench.generators.db_manager import DatabaseManager
from interfacebench.generators.db_schema import SlabRecord, InterfaceRecord
from interfacebench.simulation.db_schema import SimulationRecord
from pymatgen.core import Structure, Lattice

class TestDBManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_db.db")
        self.db_manager = DatabaseManager(self.db_path)
        
    def tearDown(self):
        self.temp_dir.cleanup()

    def _get_dummy_struct_dict(self, a=3.0, element="Cu"):
        # Create a simple cubic structure with 1 atom.
        # Different elements will make structures distinct for StructureMatcher
        return Structure(Lattice.cubic(a), [element], [[0, 0, 0]]).as_dict()

    def test_deduplicate_and_save_slabs(self):
        slab1 = {
            'base_name': 'BaO',
            'miller': '111',
            'termination': 'O',
            'natoms': 1,
            'structure_dict': self._get_dummy_struct_dict(3.0, element="Cu")
        }
        
        slab2 = {
            'base_name': 'BaO',
            'miller': '111',
            'termination': 'Ba',
            'natoms': 1,
            'structure_dict': self._get_dummy_struct_dict(3.0, element="Cu") # exact match to slab1
        }
        
        slab3 = {
            'base_name': 'BaO',
            'miller': '111',
            'termination': 'Ba',
            'natoms': 1,
            'structure_dict': self._get_dummy_struct_dict(4.0, element="Ag") # distinct
        }
        
        self.db_manager.deduplicate_and_save_slabs([slab1, slab2, slab3])
        
        with self.db_manager.SessionLocal() as session:
            records = session.query(SlabRecord).all()
            self.assertEqual(len(records), 2) # slab1 and slab3 saved, slab2 dropped
            self.assertEqual(records[0].termination, 'O')
            self.assertEqual(records[1].termination, 'Ba')

    def test_deduplicate_and_save_interfaces(self):
        struct_dict_1 = self._get_dummy_struct_dict(3.0, element="Cu")
        struct_dict_2 = self._get_dummy_struct_dict(4.0, element="Ag")

        with self.db_manager.SessionLocal() as session:
            slab_film = SlabRecord(base_name='BaO', miller='111', termination='O', natoms=1, structure_dict=struct_dict_1)
            slab_subs = SlabRecord(base_name='CuO', miller='111', termination='Cu', natoms=1, structure_dict=struct_dict_2)
            session.add_all([slab_film, slab_subs])
            session.commit()
            
            existing_record = InterfaceRecord(
                film_id=slab_film.id, subs_id=slab_subs.id,
                natoms=2, structure_dict=struct_dict_1
            )
            session.add(existing_record)
            session.commit()

        interface1 = {
            'film_name': 'BaO', 'subs_name': 'CuO',
            'film_miller': '111', 'subs_miller': '111',
            'film_termination': 'O', 'subs_termination': 'Cu',
            'natoms': 2, 'structure_dict': self._get_dummy_struct_dict(5.0, element="Au")
        }
        interface2 = {
            'film_name': 'BaO', 'subs_name': 'CuO',
            'film_miller': '111', 'subs_miller': '111',
            'film_termination': 'O', 'subs_termination': 'Cu',
            'natoms': 2, 'structure_dict': struct_dict_1 # duplicate of existing DB record
        }

        self.db_manager.deduplicate_and_save_interfaces([interface1, interface2])
        
        with self.db_manager.SessionLocal() as session:
            records = session.query(InterfaceRecord).all()
            self.assertEqual(len(records), 2) # existing + interface1

if __name__ == '__main__':
    unittest.main()