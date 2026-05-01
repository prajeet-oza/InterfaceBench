import os
import numpy as np
from pymatgen.core import Structure
from ase.io import read
from .metrics_utils import *

def load_final_structure(directory: str, engine: str, dft_file: str = "CONTCAR", ase_file: str = "relax.traj"):
    """Helper to find the final structure for a given engine."""
    if engine == "vasp":
        path = os.path.join(directory, dft_file)
        if os.path.exists(path):
            return Structure.from_file(path)
            
    elif engine == "ase":
        traj_path = os.path.join(directory, ase_file)
        if os.path.exists(traj_path):
            atoms = read(traj_path, index="-1")
            return Structure.from_ase_atoms(atoms)
            
    return None

def load_entire_trajectory(directory: str, engine: str, dft_file: str = "XDATCAR", ase_file: str = "md.traj"):
    """Helper to extract the entire trajectory for a given engine."""
    if engine == "vasp":
        path = os.path.join(directory, dft_file)
        if os.path.exists(path):
            return read(path, index=':', format="vasp-xdatcar")
            
    elif engine == "ase":
        traj_path = os.path.join(directory, ase_file)
        if os.path.exists(traj_path):
            return read(traj_path, index=":", format="traj")
            
    return None

def load_final_energy(directory: str, engine: str, dft_file: str = "OUTCAR", ase_file: str = "relax.traj"):
    """Helper to find the final energy for a given engine."""
    if engine == "vasp":
        path = os.path.join(directory, dft_file)
        if os.path.exists(path):
            atoms = read(path)
            return atoms.get_potential_energy()
            
    elif engine == "ase":
        traj_path = os.path.join(directory, ase_file)
        if os.path.exists(traj_path):
            atoms = read(traj_path, index="-1")
            return atoms.get_potential_energy()
            
    return None

# Type 1 Metric: Computed separately and stored together for comparison
def energy_metric(baseline_dir: str, test_dir: str, use_bulk_ref: bool = False, **kwargs) -> dict:
    dft_struct = load_final_structure(baseline_dir, engine="vasp")
    ase_struct = load_final_structure(test_dir, engine="ase")

    dft_energy = load_final_energy(baseline_dir, engine="vasp")
    ase_energy = load_final_energy(test_dir, engine="ase")
    
    if dft_energy is None or ase_energy is None:
        return {"error": "Missing energy files"}
    
    interface = kwargs.get('interface')
    if not interface:
        return {"error": "Missing 'interface' DB record in kwargs to compute multipliers"}

    # 1. Fetch reference slab structures from the database linked records
    film_struct = Structure.from_dict(interface.film.structure_dict) if isinstance(interface.film.structure_dict, dict) else interface.film.structure_dict
    subs_struct = Structure.from_dict(interface.subs.structure_dict) if isinstance(interface.subs.structure_dict, dict) else interface.subs.structure_dict

    # 2. Compute multipliers
    film_mult, subs_mult = compute_multipliers(dft_struct, film_struct, subs_struct)

    if use_bulk_ref:
        # 3. Retrieve bulk directories from kwargs, load energies, extract structures to calculate atom scaling
        dft_film_bulk_dir = kwargs.get('dft_film_bulk_dir')
        dft_subs_bulk_dir = kwargs.get('dft_subs_bulk_dir')
        ase_film_bulk_dir = kwargs.get('ase_film_bulk_dir')
        ase_subs_bulk_dir = kwargs.get('ase_subs_bulk_dir')

        if None in [dft_film_bulk_dir, dft_subs_bulk_dir, ase_film_bulk_dir, ase_subs_bulk_dir]:
            return {"error": "Missing one or more bulk simulation directories in kwargs"}
            
        dft_film_bulk_energy = load_final_energy(dft_film_bulk_dir, engine="vasp")
        dft_subs_bulk_energy = load_final_energy(dft_subs_bulk_dir, engine="vasp")
        ase_film_bulk_energy = load_final_energy(ase_film_bulk_dir, engine="ase")
        ase_subs_bulk_energy = load_final_energy(ase_subs_bulk_dir, engine="ase")
        
        if None in [dft_film_bulk_energy, dft_subs_bulk_energy, ase_film_bulk_energy, ase_subs_bulk_energy]:
            return {"error": "Missing energy file for one or more bulk simulations"}
            
        dft_film_bulk_struct = load_final_structure(dft_film_bulk_dir, engine="vasp")
        dft_subs_bulk_struct = load_final_structure(dft_subs_bulk_dir, engine="vasp")
        ase_film_bulk_struct = load_final_structure(ase_film_bulk_dir, engine="ase")
        ase_subs_bulk_struct = load_final_structure(ase_subs_bulk_dir, engine="ase")

        if None in [dft_film_bulk_struct, dft_subs_bulk_struct, ase_film_bulk_struct, ase_subs_bulk_struct]:
            return {"error": "Missing structure file for one or more bulk simulations"}
            
        dft_film_scaling = len(film_struct) / len(dft_film_bulk_struct)
        dft_subs_scaling = len(subs_struct) / len(dft_subs_bulk_struct)
        ase_film_scaling = len(film_struct) / len(ase_film_bulk_struct)
        ase_subs_scaling = len(subs_struct) / len(ase_subs_bulk_struct)
        
        dft_int_energy = compute_formation_energy_for_interfaces(dft_energy, dft_film_bulk_energy, dft_subs_bulk_energy, film_mult, subs_mult, dft_film_scaling, dft_subs_scaling)
        ase_int_energy = compute_formation_energy_for_interfaces(ase_energy, ase_film_bulk_energy, ase_subs_bulk_energy, film_mult, subs_mult, ase_film_scaling, ase_subs_scaling)

    else:
        # 3. Retrieve slab directories from kwargs, then load energies
        dft_film_dir = kwargs.get('dft_film_dir')
        dft_subs_dir = kwargs.get('dft_subs_dir')
        ase_film_dir = kwargs.get('ase_film_dir')
        ase_subs_dir = kwargs.get('ase_subs_dir')

        if None in [dft_film_dir, dft_subs_dir, ase_film_dir, ase_subs_dir]:
            return {"error": "Missing one or more reference slab simulation directories in kwargs"}
            
        dft_film_energy = load_final_energy(dft_film_dir, engine="vasp")
        dft_subs_energy = load_final_energy(dft_subs_dir, engine="vasp")
        ase_film_energy = load_final_energy(ase_film_dir, engine="ase")
        ase_subs_energy = load_final_energy(ase_subs_dir, engine="ase")
        
        if None in [dft_film_energy, dft_subs_energy, ase_film_energy, ase_subs_energy]:
            return {"error": "Missing energy file for one or more slab simulations"}

        dft_int_energy = compute_interfacial_energy(dft_energy, dft_film_energy, dft_subs_energy, film_mult, subs_mult)
        ase_int_energy = compute_interfacial_energy(ase_energy, ase_film_energy, ase_subs_energy, film_mult, subs_mult)

    return {
        "dft_interfacial_energy": float(dft_int_energy),
        "ase_interfacial_energy": float(ase_int_energy),
        "energy_error": float(abs(dft_int_energy - ase_int_energy))
    }


# Type 2 Metric: Computed jointly between the two trajectories
def rms_distance_metric(baseline_dir: str, test_dir: str, **kwargs) -> dict:
    dft_struct = load_final_structure(baseline_dir, engine="vasp")
    ase_struct = load_final_structure(test_dir, engine="ase")
        
    if not dft_struct or not ase_struct:
        return {"error": "Missing structure files"}
    
    rms, max_dist = compute_rms_dist(dft_struct, ase_struct, return_max=True)
    return {"rms_dist": float(rms), "max_dist": float(max_dist)}

# Type 1 Metric: Computed separately and stored together for comparison
def gaps_and_vacuum_metric(baseline_dir: str, test_dir: str, tol: float = 1e-8, **kwargs) -> dict:
    dft_struct = load_final_structure(baseline_dir, engine="vasp")
    ase_struct = load_final_structure(test_dir, engine="ase")
    
    if not dft_struct or not ase_struct:
        return {"error": "Missing structure files"}
        
    dft_v, dft_maxg, dft_ming = compute_gap_bw_layers(dft_struct, tol)
    ase_v, ase_maxg, ase_ming = compute_gap_bw_layers(ase_struct, tol)
    
    return {
        "dft_vacuum": float(dft_v) if dft_v is not None else None,
        "dft_max_gap": float(dft_maxg) if dft_maxg is not None else None,
        "ase_vacuum": float(ase_v) if ase_v is not None else None,
        "ase_max_gap": float(ase_maxg) if ase_maxg is not None else None,
        "vacuum_error": float(abs(dft_v - ase_v)) if (dft_v is not None and ase_v is not None) else None
    }

# Type 1/2 Metric: Coordination Number comparison
def coord_number_metric(baseline_dir: str, test_dir: str, **kwargs) -> dict:
    dft_struct = load_final_structure(baseline_dir, engine="vasp")
    ase_struct = load_final_structure(test_dir, engine="ase")
    
    if not dft_struct or not ase_struct:
        return {"error": "Missing structure files"}
        
    dft_cns = np.array(compute_coordination_number(dft_struct))
    ase_cns = np.array(compute_coordination_number(ase_struct))
    
    errs = ase_cns - dft_cns
    return {
        "mae": float(np.max(np.abs(errs))),
        "avg_error": float(np.mean(errs)),
        "rmse": float(np.sqrt(np.mean(errs**2)))
    }

def rdf_divergence_metric(baseline_dir: str, test_dir: str, rmax: float = 30.0, dr: float = 0.5, **kwargs) -> dict:
    dft_path = os.path.join(baseline_dir, "XDATCAR")
    ase_path = os.path.join(test_dir, "md.traj")
    
    if not os.path.exists(dft_path) or not os.path.exists(ase_path):
        return {"error": "Missing trajectory files (XDATCAR or md.traj)"}
        
    dft_traj = read(dft_path, index=':', format="vasp-xdatcar")
    _, dft_rdf = compute_rdf_for_trajectory(dft_traj, rmax, dr)
    
    ase_traj = read(ase_path, index=':')
    _, ase_rdf = compute_rdf_for_trajectory(ase_traj, rmax, dr)
    
    return {
        "hellinger_distance": float(compute_rdf_hellinger_distance(dft_rdf, ase_rdf)),
        "kl_divergence": float(compute_rdf_kl_divergence(dft_rdf, ase_rdf))
    }

def min_distance(baseline_dir: str, test_dir: str, **kwargs) -> dict:
    dft_traj = load_entire_trajectory(baseline_dir, engine="vasp")
    ase_traj = load_entire_trajectory(test_dir, engine="ase")
    
    if not dft_traj or not ase_traj:
        return {"error": "Missing trajectory files"}
    else:
        return {'dft_min_distance': compute_min_distance(dft_traj, **kwargs),
                'ase_min_distance': compute_min_distance(ase_traj, **kwargs)}

def dynamic_knn_preservation(baseline_dir: str, test_dir: str, **kwargs) -> dict:
    dft_traj = load_entire_trajectory(baseline_dir, engine="vasp")
    ase_traj = load_entire_trajectory(test_dir, engine="ase")

    if not dft_traj or not ase_traj:
        return {"error": "Missing trajectory files"}
    else:
        return {'dft_dynamic_knn': compute_dynamic_knn_preservation(dft_traj, **kwargs),
                'ase_dynamic_knn': compute_dynamic_knn_preservation(ase_traj, **kwargs)}

def large_velocity(baseline_dir: str, test_dir: str, **kwargs) -> dict:
    dft_traj = load_entire_trajectory(baseline_dir, engine="vasp")
    ase_traj = load_entire_trajectory(test_dir, engine="ase")

    if not dft_traj or not ase_traj:
        return {"error": "Missing trajectory files"}
    else:
        return {'dft_large_velocity': compute_large_velocity(dft_traj, **kwargs),
                'ase_large_velocity': compute_large_velocity(ase_traj, **kwargs)}

def lindemann_index(baseline_dir: str, test_dir: str, **kwargs) -> dict:
    dft_traj = load_entire_trajectory(baseline_dir, engine="vasp")
    ase_traj = load_entire_trajectory(test_dir, engine="ase")

    if not dft_traj or not ase_traj:
        return {"error": "Missing trajectory files"}
    else:
        return {'dft_lindemann_index': compute_lindemann_index(dft_traj, **kwargs),
                'ase_lindemann_index': compute_lindemann_index(ase_traj, **kwargs)}

def velocity_distribution_divergence(baseline_dir: str, test_dir: str, **kwargs) -> dict:
    dft_traj = load_entire_trajectory(baseline_dir, engine="vasp")
    ase_traj = load_entire_trajectory(test_dir, engine="ase")

    if not dft_traj or not ase_traj:
        return {"error": "Missing trajectory files"}
    else:
        return {'dft_velocity_distribution': compute_velocity_distribution_divergence(dft_traj, **kwargs),
                'ase_velocity_distribution': compute_velocity_distribution_divergence(ase_traj, **kwargs)}

def velocity_skewness(baseline_dir: str, test_dir: str, **kwargs) -> dict:
    dft_traj = load_entire_trajectory(baseline_dir, engine="vasp")
    ase_traj = load_entire_trajectory(test_dir, engine="ase")

    if not dft_traj or not ase_traj:
        return {"error": "Missing trajectory files"}
    else:
        return {'dft_velocity_skewness': compute_velocity_skewness(dft_traj, **kwargs),
                'ase_velocity_skewness': compute_velocity_skewness(ase_traj, **kwargs)}


def register_default_metrics(orchestrator):
    orchestrator.register_metric("interface_energy", "relax", energy_metric)
    orchestrator.register_metric("formation_energy", "relax", energy_metric, use_bulk_ref=True)
    orchestrator.register_metric("rms_distance", "relax", rms_distance_metric)
    orchestrator.register_metric("gaps_and_vacuum", "relax", gaps_and_vacuum_metric)
    orchestrator.register_metric("coordination_number", "relax", coord_number_metric)
    orchestrator.register_metric("rdf_divergence", "relax", rdf_divergence_metric)
    orchestrator.register_metric("min_distance", "md", min_distance, use_half_initial=True, enable_tqdm=True)
    orchestrator.register_metric("static_knn_preservation", "md", dynamic_knn_preservation, scale=0.4, n_neighbors=6, enable_tqdm=True)
    orchestrator.register_metric("large_velocity", "md", large_velocity, enable_tqdm=True, temperature=300.0)
    orchestrator.register_metric("lindemann_index", "md", lindemann_index, local_threshold=0.2, global_threshold=0.15, enable_tqdm=True)
    orchestrator.register_metric("velocity_distribution_divergence", "md", velocity_distribution_divergence, temperature=300.0, threshold=0.2, start_at_frame=5000, n_points=15000)
    orchestrator.register_metric("velocity_skewness", "md", velocity_skewness, temperature=300.0, start_at_frame=5000, n_points=15000)