import logging
from pathlib import Path
import json
from pymatgen.core import Structure

from interfacebench.generators.db_manager import DatabaseManager
from interfacebench.generators.db_schema import InterfaceRecord
from interfacebench.generators.db_schema import SlabRecord
from interfacebench.generators.db_schema import BulkRecord
from .db_schema import SimulationRecord

logger = logging.getLogger(__name__)

class AseScheduler:
    def __init__(self, db_path: str | Path, work_dir: str | Path):
        self.db_path = Path(db_path).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.db_manager = DatabaseManager(str(self.db_path))
        
        # Ensure the simulations table is bound safely to the existing engine
        SimulationRecord.__table__.create(self.db_manager.engine, checkfirst=True)

    def generate_inputs(self, models: list, sim_type: str, write_run_script: bool = True, bulk_cif_dir: str | Path | None = None, **kwargs):
        """
        Prepares directories, POSCARs, and scripts for ASE calculations so users 
        can run them later (or on a cluster).
        models: [{'tag': 'mace_mpa', 'model_code': "MACECalculator(model='medium')", 'model_import': 'from mace.calculators import MACECalculator'}, ...]
        """
        with self.db_manager.SessionLocal() as session:
            
            # 1. Schedule Interfaces
            interfaces = session.query(InterfaceRecord).all()
            
            for record in interfaces:
                struct_data = record.structure_dict
                struct = struct_data if isinstance(struct_data, Structure) else Structure.from_dict(struct_data)
                cif_name = f"{record.film.base_name}_{record.subs.base_name}_{record.id}"
                
                for model_dict in models:
                    tag = model_dict['tag']
                    
                    existing = session.query(SimulationRecord).filter_by(
                        interface_id=record.id, engine="ase", sim_type=sim_type, model_tag=tag
                    ).first()
                    
                    if existing:
                        logger.debug(f"Skipping {cif_name} with {tag}, already configured.")
                        continue
                        
                    calc_folder = self.work_dir / f"ase_{sim_type}_{cif_name}_{tag}"
                    calc_folder.mkdir(parents=True, exist_ok=True)
                    
                    # Write structure for ASE to read
                    struct.to(filename=str(calc_folder / "POSCAR"))
                    
                    if write_run_script:
                        self._write_ase_run_script(calc_folder, sim_type, model_dict, kwargs)

                    sim_record = SimulationRecord(
                        interface_id=record.id, engine="ase", sim_type=sim_type, 
                        model_tag=tag, status="pending", calc_folder=str(calc_folder)
                    )
                    session.add(sim_record)
                    logger.info(f"Generated ASE {sim_type} inputs for {cif_name} with {tag}")
            
            # 2. Schedule Slabs
            slabs = session.query(SlabRecord).all()
            
            for record in slabs:
                struct_data = record.structure_dict
                struct = struct_data if isinstance(struct_data, Structure) else Structure.from_dict(struct_data)
                cif_name = f"slab_{record.base_name}_{record.miller}_{record.id}"
                
                for model_dict in models:
                    tag = model_dict['tag']
                    
                    existing = session.query(SimulationRecord).filter_by(
                        slab_id=record.id, engine="ase", sim_type=sim_type, model_tag=tag
                    ).first()
                    
                    if existing:
                        continue
                        
                    calc_folder = self.work_dir / f"ase_{sim_type}_{cif_name}_{tag}"
                    calc_folder.mkdir(parents=True, exist_ok=True)
                    
                    struct.to(filename=str(calc_folder / "POSCAR"))
                    if write_run_script:
                        self._write_ase_run_script(calc_folder, sim_type, model_dict, kwargs)

                    sim_record = SimulationRecord(
                        slab_id=record.id, engine="ase", sim_type=sim_type, 
                        model_tag=tag, status="pending", calc_folder=str(calc_folder)
                    )
                    session.add(sim_record)
                    logger.info(f"Generated ASE {sim_type} inputs for slab {cif_name} with {tag}")

            # 3. Schedule Bulks
            if sim_type == "relax":
                bulks = session.query(BulkRecord).all()
                
                for bulk_record in bulks:
                    b_name = bulk_record.name
                    struct_data = bulk_record.structure_dict
                    struct = struct_data if isinstance(struct_data, Structure) else Structure.from_dict(struct_data)
                    
                    for model_dict in models:
                        tag = model_dict['tag']
                        existing = session.query(SimulationRecord).filter_by(
                            bulk_name=b_name, engine="ase", sim_type=sim_type, model_tag=tag
                        ).first()
                        
                        if existing:
                            continue
                            
                        calc_folder = self.work_dir / f"ase_{sim_type}_bulk_{b_name}_{bulk_record.id}_{tag}"
                        calc_folder.mkdir(parents=True, exist_ok=True)
                        struct.to(filename=str(calc_folder / "POSCAR"))
                        
                        if write_run_script:
                            bulk_kwargs = kwargs.copy()
                            bulk_kwargs['volume_relax'] = bulk_kwargs.get('volume_relax', True)  # Force cell filter for bulk
                            self._write_ase_run_script(calc_folder, sim_type, model_dict, bulk_kwargs)

                        sim_record = SimulationRecord(
                            bulk_name=b_name, engine="ase", sim_type=sim_type, 
                            model_tag=tag, status="pending", calc_folder=str(calc_folder)
                        )
                        session.add(sim_record)
                        logger.info(f"Generated ASE {sim_type} inputs for bulk {b_name} with {tag}")

            session.commit()

    def _write_ase_run_script(self, calc_folder: Path, sim_type: str, model_dict: dict, kwargs: dict):
        # Dump the config so the standalone script can use it
        with open(calc_folder / "kwargs_config.json", "w") as f:
            json.dump(kwargs, f)
        
        model_code = model_dict.get('model_code', 'None # Replace with your calculator')
        model_import = model_dict.get('model_import', '# Define your calculator import here')
        script_content = f"""import json
from ase.io import read, Trajectory
from ase.md.logger import MDLogger
from ase.filters import UnitCellFilter
from ase.md.nose_hoover_chain import NoseHooverChainNVT
import ase.optimize
{model_import}

# Load structure
atoms = read("POSCAR")

# Load calculator
atoms.calc = {model_code}

# Load kwargs
with open("kwargs_config.json", "r") as f:
    kwargs = json.load(f)
"""
        
        if sim_type == "relax":
            script_content += """
opt_name = kwargs.get('optimizer', 'BFGS')
optimizer_cls = getattr(ase.optimize, opt_name)
if kwargs.get('volume_relax'):
    atoms = UnitCellFilter(atoms)
relax = optimizer_cls(atoms, trajectory='relax.traj', logfile='relax.log')
relax.run(fmax=kwargs.get('fmax', 0.05), steps=kwargs.get('maxstep', 1000))
"""

        elif sim_type == "md":
            script_content += """
timestep = kwargs.get('timestep', 1.0)
temperature = kwargs.get('temperature', 300)
dyn = NoseHooverChainNVT(atoms, timestep=timestep, temperature_K=temperature, tdamp=timestep*100)
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=True, mode='w'))
traj = Trajectory('md.traj', 'w', atoms)
dyn.attach(traj.write, interval=kwargs.get('traj_interval', 10))
dyn.run(steps=kwargs.get('numsteps', 1000))
"""
        (calc_folder / "run_ase.py").write_text(script_content)