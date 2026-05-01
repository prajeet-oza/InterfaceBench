import argparse
import logging
import ast

from .pipeline import InterfaceBenchPipeline

def parse_models(models_str):
    """Parses a comma-separated list of tags OR a raw string representation of a list of dicts"""
    try:
        if models_str.startswith('['):
            return ast.literal_eval(models_str)
        else:
            tags = models_str.split(',')
            return [{'tag': t.strip(), 'model_code': 'None', 'model_import': '# Define your calculator import here'} for t in tags]
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid models format: {e}")

def main():
    parser = argparse.ArgumentParser(description="InterfaceBench Orchestrator CLI")
    parser.add_argument('--step', type=str, choices=['all', 'generate', 'dft', 'mlip', 'evaluate'], default='all',
                        help="Pipeline step to run.")
    parser.add_argument('--db_path', type=str, default='structures.db', help="Path to the SQLite database")
    parser.add_argument('--work_dir', type=str, default='simulations', help="Directory to store HPC simulation files")
    parser.add_argument('--input_source', type=str, default='inputs.csv', help="Path to CSV/YAML for structure generation")
    parser.add_argument('--models', type=parse_models, help="Comma-separated model tags or JSON string of dicts")
    parser.add_argument('--sim_type', type=str, choices=['relax', 'md'], default='relax', help="Simulation type (relax or md)")
    parser.add_argument('--log_level', type=str, default='INFO', help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    pipeline = InterfaceBenchPipeline(
        db_path=args.db_path,
        work_dir=args.work_dir
    )
    
    if args.step == 'all':
        pipeline.run_all(input_source=args.input_source, models=args.models, sim_type=args.sim_type)
    elif args.step == 'generate':
        pipeline.generate_structures(input_source=args.input_source)
    elif args.step == 'dft':
        pipeline.create_dft_tasks(sim_type=args.sim_type)
    elif args.step == 'mlip':
        pipeline.simulate_mlip_tasks(models=args.models, sim_type=args.sim_type)
    elif args.step == 'evaluate':
        pipeline.compute_metrics()

if __name__ == "__main__":
    main()