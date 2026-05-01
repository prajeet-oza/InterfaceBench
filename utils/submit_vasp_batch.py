import argparse
import shutil
import subprocess
from pathlib import Path
from interfacebench.generators.db_manager import DatabaseManager
from interfacebench.simulation.db_schema import SimulationRecord

def submit_vasp_jobs(db_path: str, job_script_path: str):
    """
    Finds all 'pending' VASP jobs in the database, copies the provided Slurm batch 
    script to their folder, and submits them to the queue using sbatch.
    """
    db_manager = DatabaseManager(db_path)
    job_script = Path(job_script_path).resolve()
    
    if not job_script.exists():
        print(f"[ERROR] Slurm job script '{job_script}' not found.")
        return
        
    with db_manager.SessionLocal() as session:
        pending_jobs = session.query(SimulationRecord).filter_by(engine="vasp", status="pending").all()
        
        if not pending_jobs:
            print("No pending VASP jobs found in the database.")
            return
            
        print(f"Found {len(pending_jobs)} pending VASP jobs. Submitting to Slurm...\n")
        
        for job in pending_jobs:
            calc_dir = Path(job.calc_folder)
            
            # Copy the submission script into the simulation directory
            target_script_path = calc_dir / job_script.name
            shutil.copy(job_script, target_script_path)
            
            # Run sbatch
            res = subprocess.run(['sbatch', job_script.name], cwd=calc_dir, capture_output=True, text=True)
            
            if res.returncode == 0:
                job.job_id = res.stdout.strip().split()[-1]
                job.status = "queue"
                print(f"-> Submitted Job ID {job.job_id} for {calc_dir.name}")
            else:
                print(f"-> [FAILED] Could not submit {calc_dir.name}: {res.stderr.strip()}")
                
        session.commit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit all pending VASP simulations to Slurm.")
    parser.add_argument("--script", type=str, required=True, help="Path to the Slurm submission script (e.g., submit.sh)")
    parser.add_argument("--db", type=str, default="structures.db", help="Path to the SQLite database")
    args = parser.parse_args()
    
    submit_vasp_jobs(args.db, args.script)