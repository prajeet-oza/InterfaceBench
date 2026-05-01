import logging
from typing import List, Dict
from pathlib import Path

from .generators.generator import GeneratorPipeline
from .simulation.vasp_scheduler import VaspScheduler
from .simulation.ase_scheduler import AseScheduler
from .metrics.orchestrator import BenchmarkOrchestrator

logger = logging.getLogger(__name__)

class InterfaceBenchPipeline:
    """
    Orchestrator for the InterfaceBench flow:
    1. Generate structures (interfaces & slabs)
    2. Create files for DFT tasks
    3. Simulate MLIP tasks
    4. Compute metrics for DFT and MLIP
    """
    
    def __init__(
        self, 
        db_path: str = 'structures.db',
        work_dir: str = 'simulations'
    ):
        self.db_path = db_path
        self.work_dir = work_dir
        
    def generate_structures(self, input_source: str):
        logger.info(f"Generating structures from {input_source}...")
        generator = GeneratorPipeline(db_path=self.db_path)
        generator.run_from_file(input_source)
        logger.info("Structure generation step completed.")
        
    def create_dft_tasks(self, sim_type: str = 'relax'):
        logger.info("Creating files for DFT tasks...")
        scheduler = VaspScheduler(db_path=self.db_path, work_dir=self.work_dir)
        scheduler.generate_inputs(sim_type=sim_type)
        logger.info("DFT tasks setup completed.")
        
    def simulate_mlip_tasks(self, models: List[Dict[str, str]], sim_type: str = 'relax'):
        logger.info(f"Simulating MLIP tasks for models: {[m['tag'] for m in models]}...")
        scheduler = AseScheduler(db_path=self.db_path, work_dir=self.work_dir)
        scheduler.generate_inputs(models=models, sim_type=sim_type)
        logger.info("MLIP simulations completed.")
        
    def compute_metrics(self):
        logger.info("Computing metrics for DFT and MLIP...")
        orchestrator = BenchmarkOrchestrator(db_path=self.db_path)
        orchestrator.run()
        logger.info("Metrics evaluation step completed.")

    def run_all(self, input_source: str, models: List[Dict[str, str]], sim_type: str = 'relax'):
        logger.info("Starting InterfaceBench full pipeline...")
        self.generate_structures(input_source)
        self.create_dft_tasks(sim_type)
        self.simulate_mlip_tasks(models, sim_type)
        self.compute_metrics()
        logger.info("Full pipeline completed successfully.")