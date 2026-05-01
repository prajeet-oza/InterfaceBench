# InterfaceBench

**InterfaceBench** is a Python-based computational workflow and evaluation framework designed to benchmark Universal Machine-Learned Interatomic Potentials (uMLIPs) on complex solid-solid interfaces. 

Solid-solid interfaces define the performance of solid-state batteries, heterostructures, and functional ceramics. While uMLIPs are widely benchmarked on bulk materials, their out-of-distribution performance on interfacial features (abrupt compositional changes, local strains, under-coordinated surface atoms) is generally untested. 

InterfaceBench solves the combinatorially large task of generating, deduplicating, and high-throughput simulating solid-solid interfaces while tracking strict thermodynamic lineage via a relational SQL database.

## Core Capabilities
* **Combinatorial Structure Generation**: Automatically slices and aligns bulk phases to generate symmetry-distinct film and substrate combinations using Coherent Interface Builders.
* **Thermodynamic Data Lineage**: Foreign-key-enforced relational SQL tracking ensures interface energies are strictly calculated against their identically-configured parent slab states. 
* **$O(N)$ Grouped Deduplication**: A fast atom-count grouping algorithm bypasses the traditional global $O(N^2)$ structure matching bottleneck.
* **Idempotent Headless Simulation**: Safe execution architecture across HPC systems using self-contained JSON/Python scripts for VASP (DFT) and ASE (MLIPs).
* **Physical Fidelity Metrics**: Extensive tracking of interfacial phenomena ranging from static CN deviations and Radial Distribution divergence to dynamic minimum distances, Lindemann indices, and Maxwell-Boltzmann thermodynamics checks.

## Installation

1. Clone the repository and navigate to the project root.
2. Install in editable mode via pip:

```bash
pip install -e .
```

## Quick Start

### 1. Input Configuration (`inputs.csv`)
Define the desired solid-solid interfaces via a CSV or YAML file. Your initial `cif` bulk geometries must exist in the working directory:
```csv
film,subs,film_miller,subs_miller,film_thickness,subs_thickness,generate_slabs,in_layers
BaO,CuO,1 0 0,1 1 1,10.0,10.0,True,True
```

### 2. Using the Command Line Interface (CLI)

You can execute the entire pipeline or specific individual steps:

**Run the entire pipeline (Generation -> DFT Setup -> MLIP Setup -> Metrics Evaluation):**
```bash
python -m interfacebench --step all \
    --input_source inputs.csv \
    --db_path structures.db \
    --work_dir simulations \
    --sim_type relax \
    --models "[{'tag': 'mattersim-5m', 'model_code': 'MatterSimCalculator()', 'model_import': 'from mattersim.forcefield import MatterSimCalculator'}]"
```

**Run specific steps:**
```bash
# Just generate the structures and push to SQL
python -m interfacebench --step generate --input_source inputs.csv

# Prepare simulation input directories for VASP (DFT)
python -m interfacebench --step dft --sim_type relax

# Generate calculation scripts for uMLIPs
python -m interfacebench --step mlip --models "[{'tag': 'mace-mpa', 'model_code': 'mace_mp(model='medium-mpa-0')', 'model_import': 'from mace.calculators import mace_mp'}, {'tag': 'mattersim-5m', 'model_code': 'MatterSimCalculator()', 'model_import': 'from mattersim.forcefield import MatterSimCalculator'}]"

# Read completed trajectories and populate the metrics schema
python -m interfacebench --step evaluate
```

### 3. Using the Python API
You can seamlessly integrate InterfaceBench blocks directly into existing research scripts:

```python
from interfacebench.pipeline import InterfaceBenchPipeline

pipeline = InterfaceBenchPipeline(
    db_path="structures.db",
    work_dir="my_simulations"
)

# Generate structures
pipeline.generate_structures("config.yaml")

# Set up MLIP tasks
models_to_test = [
    {"tag": "mace-mpa", "model_code": "mace_mp(model='medium-mpa-0')", "model_import": "from mace.calculators import mace_mp"},
    {"tag": "mattersim", "model_code": "MatterSimCalculator()", "model_import": "from mattersim.forcefield import MatterSimCalculator"}
]
pipeline.simulate_mlip_tasks(models=models_to_test, sim_type="md")

# ... (Run jobs on cluster) ...

# Run custom/default evaluations automatically against the database
pipeline.compute_metrics()
```

## Metric Output Interpretation

Metrics are written back to the `metrics` table in the SQLite database linked directly to the parent `SimulationRecord`. This makes it incredibly easy to use pandas/SQLAlchemy to plot comparisons.
