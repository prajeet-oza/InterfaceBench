import argparse
import subprocess
from pathlib import Path
from interfacebench.generators.db_manager import DatabaseManager
from interfacebench.simulation.db_schema import SimulationRecord

def execute_ase_jobs(db_path: str):
    """
    Finds all 'pending' ASE jobs in the database and executes them locally one by one.
    Updates the database with 'running', 'complete', or 'aborted' statuses.
    """
    db_manager = DatabaseManager(db_path)
    
    with db_manager.SessionLocal() as session:
        pending_jobs = session.query(SimulationRecord).filter_by(engine="ase", status="pending").all()
        
        if not pending_jobs:
            print("No pending ASE jobs found in the database.")
            return
            
        print(f"Found {len(pending_jobs)} pending ASE jobs. Starting execution...\n")
        
        for job in pending_jobs:
            calc_dir = Path(job.calc_folder)
            script_path = calc_dir / "run_ase.py"
            
            if not script_path.exists():
                print(f"[ERROR] Script not found for job {job.id} at {script_path}")
                continue
                
            print(f"-> Executing job {job.id} in {calc_dir.name}...")
            
            # Mark as running
            job.status = "running"
            session.commit()
            
            # Execute the script
            res = subprocess.run(["python", "run_ase.py"], cwd=calc_dir)
            
            # Update status based on exit code
            if res.returncode == 0:
                job.status = "complete"
                print(f"   [SUCCESS] Job {job.id} finished successfully.\n")
            else:
                job.status = "aborted"
                print(f"   [FAILED] Job {job.id} aborted with exit code {res.returncode}.\n")
                
            session.commit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute all pending ASE simulations locally.")
    parser.add_argument("--db", type=str, default="structures.db", help="Path to the SQLite database")
    args = parser.parse_args()
    
    execute_ase_jobs(args.db)