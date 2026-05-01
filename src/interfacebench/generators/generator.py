import logging
from pathlib import Path
import numpy as np

from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.core.structure import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder

logger = logging.getLogger(__name__)

class SlabBuilder:
    def __init__(self, slab_base, miller, base_structure_loc=".", termination="no_term"):
        self.miller = miller
        self.termination = termination
        self.base_structure_loc = Path(base_structure_loc)
        
        if isinstance(slab_base, Structure):
            self.slab_base_name = slab_base.composition.reduced_formula
            self.base_structure = slab_base
        else:
            self.slab_base_name = slab_base
            cif_path = self.base_structure_loc / f"{self.slab_base_name}.cif"
            self.base_structure = Structure.from_file(cif_path)

    def generate(self, slab_thickness: float, shift: float = 0.0, min_vacuum_size: float = 5.0, in_layers: bool = True) -> dict:
        """Generates the slab and returns a data dictionary instead of saving to disk."""
        slab_gen = SlabGenerator(
            self.base_structure, self.miller, slab_thickness,
            min_vacuum_size=min_vacuum_size, in_unit_planes=in_layers
        )
        struct = slab_gen.get_slab(shift=shift)
        
        return {
            'base_name': self.slab_base_name,
            'miller': "".join(map(str, self.miller)),
            'termination': self.termination,
            'natoms': len(struct),
            'structure_dict': struct.as_dict()
        }

class InterfaceBuilder:
    def __init__(self, zsl_generator, base_structure_loc=".", threshold=100):
        self.zsl = zsl_generator
        self.base_structure_loc = Path(base_structure_loc)
        self.threshold = threshold

    def generate_interfaces(self, film: str, subs: str, film_miller: tuple, subs_miller: tuple, 
                          film_thickness: float, subs_thickness: float, gap: float = 2.0, vacuum: float = 20.0,
                          generate_slabs: bool = True, in_layers: bool = True):
        """Yields interface dictionaries as they are generated."""
        film_structure = Structure.from_file(self.base_structure_loc / f"{film}.cif")
        subs_structure = Structure.from_file(self.base_structure_loc / f"{subs}.cif")
        
        try:
            cib = CoherentInterfaceBuilder(
                film_structure=film_structure, substrate_structure=subs_structure,
                film_miller=film_miller, substrate_miller=subs_miller, zslgen=self.zsl
            )
        except Exception as e:
            logger.error(f"Failed building coherent interface: {e}")
            return

        for termination in cib.terminations:
            try:
                ints = list(cib.get_interfaces(
                    termination=termination, gap=gap, vacuum_over_film=vacuum,
                    film_thickness=film_thickness, substrate_thickness=subs_thickness,
                    in_layers=in_layers
                ))
                
                if not ints or len(ints[0]) > self.threshold:
                    continue
                    
                struct = ints[0]
                
                # Translate structure to z=0
                z_min = np.min(struct.frac_coords[:, 2])
                struct.translate_sites(np.arange(struct.num_sites), np.array([0, 0, -z_min]))
                
                # Extract the exact shifts used for these terminations from the cib internal slabs
                # film_shift = cib._film_slabs[termination[0]].shift
                # subs_shift = cib._sub_slabs[termination[1]].shift
                
                if generate_slabs:
                    # Dynamically generate the corresponding parent slabs with the correct thickness
                    film_builder = SlabBuilder(slab_base=film_structure, miller=film_miller, termination=termination[0])
                    film_builder.slab_base_name = film  # Ensure exact name match for DB linkage
                    film_slab_dict = film_builder.generate(slab_thickness=film_thickness, min_vacuum_size=vacuum, in_layers=in_layers)
                    
                    subs_builder = SlabBuilder(slab_base=subs_structure, miller=subs_miller, termination=termination[1])
                    subs_builder.slab_base_name = subs  # Ensure exact name match for DB linkage
                    subs_slab_dict = subs_builder.generate(slab_thickness=subs_thickness, min_vacuum_size=vacuum, in_layers=in_layers)
                else:
                    film_slab_dict = None
                    subs_slab_dict = None

                yield {
                    'film_name': film,
                    'subs_name': subs,
                    'film_miller': "".join(map(str, film_miller)),
                    'subs_miller': "".join(map(str, subs_miller)),
                    'film_termination': termination[0],
                    'subs_termination': termination[1],
                    'natoms': len(struct),
                    'structure_dict': struct.as_dict(),
                    'film_slab': film_slab_dict,
                    'subs_slab': subs_slab_dict,
                    'metadata_dict': {
                        'film_termination': termination[0],
                        'subs_termination': termination[1],
                        'film_miller': film_miller,
                        'subs_miller': subs_miller
                    }
                }
                
            except RuntimeError as e: # Catching specific expected math/generation errors
                logger.error(f"Generation failed for termination {termination}: {e}")

class GeneratorPipeline:
    """
    A high-level pipeline to generate and save interfaces (and optionally slabs) with minimal code.
    """
    def __init__(self, db_path="structures.db", base_structure_loc=".", zsl: ZSLGenerator | None = None, threshold: int = 1000):
        from .db_manager import DatabaseManager
            
        self.db_manager = DatabaseManager(db_path=db_path)
        self.zsl = zsl or ZSLGenerator()
        self.interface_builder = InterfaceBuilder(zsl_generator=self.zsl, base_structure_loc=base_structure_loc, threshold=threshold)

    def run(self, film: str, subs: str, film_miller: tuple, subs_miller: tuple,
            film_thickness: float = 10.0, subs_thickness: float = 10.0,
            gap: float = 2.0, vacuum: float = 20.0, generate_slabs: bool = True, in_layers: bool = True):
        
        logger.info(f"Starting generation pipeline for {film} and {subs}...")
        interfaces = list(self.interface_builder.generate_interfaces(
            film=film, subs=subs, film_miller=film_miller, subs_miller=subs_miller,
            film_thickness=film_thickness, subs_thickness=subs_thickness, gap=gap, vacuum=vacuum,
            generate_slabs=generate_slabs, in_layers=in_layers
        ))
        
        if not interfaces:
            logger.warning("No interfaces generated.")
            return

        if generate_slabs:
            slabs = []
            for interface in interfaces:
                if interface['film_slab']: slabs.append(interface['film_slab'])
                if interface['subs_slab']: slabs.append(interface['subs_slab'])
            self.db_manager.deduplicate_and_save_slabs(slabs)

        self.db_manager.deduplicate_and_save_interfaces(interfaces)
        logger.info(f"Successfully generated and saved {len(interfaces)} interfaces.")

    def run_from_file(self, file_path: str):
        """
        Reads generation parameters from a CSV or YAML file and executes the pipeline iteratively.
        """
        file_path_obj = Path(file_path)
        if file_path_obj.suffix.lower() == '.csv':
            import pandas as pd
            df = pd.read_csv(file_path_obj)
            for _, row in df.iterrows():
                # drop nan values to allow using default args if left blank in CSV
                kwargs = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
                
                # Parse miller indices if they are strings e.g., "1 0 0" or "1,0,0"
                for key in ['film_miller', 'subs_miller']:
                    if key in kwargs and isinstance(kwargs[key], str):
                        kwargs[key] = tuple(map(int, kwargs[key].replace(',', ' ').split()))
                        
                self.run(**kwargs)
                
        elif file_path_obj.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            with open(file_path_obj, 'r') as f:
                configs = yaml.safe_load(f)
            for config in configs:
                # miller indices in yaml might be lists, convert to tuple
                for key in ['film_miller', 'subs_miller']:
                    if key in config and isinstance(config[key], list):
                        config[key] = tuple(config[key])
                        
                self.run(**config)
        else:
            raise ValueError(f"Unsupported file format: {file_path_obj.suffix}. Please use .csv or .yaml")