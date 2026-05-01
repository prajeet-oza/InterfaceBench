import subprocess
import re
import logging
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPMDSet, MPRelaxSet
from pymatgen.io.vasp import Incar

# Import your DB setup
from interfacebench.generators.db_manager import DatabaseManager
from interfacebench.generators.db_schema import InterfaceRecord
from interfacebench.generators.db_schema import SlabRecord
from .db_schema import SimulationRecord

logger = logging.getLogger(__name__)

class VaspScheduler:
    def __init__(self, db_path: str | Path, work_dir: str | Path):
        self.db_path = Path(db_path).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.db_manager = DatabaseManager(str(self.db_path))
        
        # Ensure the simulations table is bound safely to the existing engine
        SimulationRecord.__table__.create(self.db_manager.engine, checkfirst=True)

    def generate_inputs(self, sim_type: str, bulk_cif_dir: str | Path | None = None, **kwargs):
        """Generates inputs for either 'relax' or 'md'."""
        with self.db_manager.SessionLocal() as session:
            
            # 1. Schedule Interfaces
            interfaces = session.query(InterfaceRecord).all()
            for record in interfaces:
                # Query if a record already exists to prevent overwriting
                existing = session.query(SimulationRecord).filter_by(
                    interface_id=record.id, engine="vasp", sim_type=sim_type
                ).first()
                if existing:
                    continue

                struct_data = record.structure_dict
                struct = struct_data if isinstance(struct_data, Structure) else Structure.from_dict(struct_data)
                
                cif_name = f"{record.film.base_name}_{record.subs.base_name}_{record.id}"
                
                # Determine folder name and Pymatgen set
                calc_folder = self.work_dir / f"vasp_{sim_type}_{cif_name}"
                calc_folder.mkdir(parents=True, exist_ok=True)
                
                if sim_type == "relax":
                    generator = MPRelaxSet(struct, user_incar_settings={'ISIF': 2, **kwargs})
                elif sim_type == "md":
                    # Extract MD specific kwargs or use defaults
                    generator = MPMDSet(struct, **kwargs) 
                
                # Write inputs
                generator.write_input(calc_folder)
                
                # Register in DB
                new_sim = SimulationRecord(
                    interface_id=record.id,
                    engine="vasp",
                    sim_type=sim_type,
                    model_tag="vasp_dft",
                    status="pending",
                    calc_folder=str(calc_folder)
                )
                session.add(new_sim)
                logger.info(f"Generated {sim_type} inputs for {cif_name}")
            
            # 2. Schedule Slabs
            slabs = session.query(SlabRecord).all()
            for record in slabs:
                existing = session.query(SimulationRecord).filter_by(
                    slab_id=record.id, engine="vasp", sim_type=sim_type
                ).first()
                if existing:
                    continue

                struct_data = record.structure_dict
                struct = struct_data if isinstance(struct_data, Structure) else Structure.from_dict(struct_data)
                
                cif_name = f"slab_{record.base_name}_{record.miller}_{record.id}"
                
                calc_folder = self.work_dir / f"vasp_{sim_type}_{cif_name}"
                calc_folder.mkdir(parents=True, exist_ok=True)
                
                if sim_type == "relax":
                    generator = MPRelaxSet(struct, user_incar_settings={'ISIF': 2, **kwargs})
                elif sim_type == "md":
                    generator = MPMDSet(struct, **kwargs)
                
                generator.write_input(calc_folder)
                
                new_sim = SimulationRecord(
                    slab_id=record.id, engine="vasp", sim_type=sim_type,
                    model_tag="vasp_dft", status="pending", calc_folder=str(calc_folder)
                )
                session.add(new_sim)
                logger.info(f"Generated {sim_type} inputs for slab {cif_name}")

            # 3. Schedule Bulks (Only meaningful for relaxations to serve as references)
            if sim_type == "relax" and bulk_cif_dir:
                bulk_cif_dir = Path(bulk_cif_dir)
                slabs = session.query(SlabRecord).all()
                bulk_names = set(record.base_name for record in slabs)
                
                for b_name in bulk_names:
                    existing = session.query(SimulationRecord).filter_by(
                        bulk_name=b_name, engine="vasp", sim_type=sim_type
                    ).first()
                    if existing:
                        continue

                    cif_path = bulk_cif_dir / f"{b_name}.cif"
                    if not cif_path.exists():
                        logger.warning(f"CIF not found for bulk {b_name} at {cif_path}")
                        continue

                    struct = Structure.from_file(cif_path)
                    calc_folder = self.work_dir / f"vasp_{sim_type}_bulk_{b_name}"
                    calc_folder.mkdir(parents=True, exist_ok=True)
                    
                    # Ensure volume changes are permitted by setting ISIF=3 for bulk
                    generator = MPRelaxSet(struct, user_incar_settings={'ISIF': 3, **kwargs})
                    generator.write_input(calc_folder)
                    
                    new_sim = SimulationRecord(
                        bulk_name=b_name, engine="vasp", sim_type=sim_type,
                        model_tag="vasp_dft", status="pending", calc_folder=str(calc_folder)
                    )
                    session.add(new_sim)
                    logger.info(f"Generated {sim_type} inputs for bulk {b_name}")

            session.commit()