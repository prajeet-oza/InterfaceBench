import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Any
from tqdm import tqdm

from scipy import stats
from scipy.stats import linregress, skew
from scipy.special import kl_div

import ase.neighborlist as nl
from ase import Atoms, units
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.local_env import CrystalNN

class MaxwellBoltzmannComputer:
    def __init__(self, traj: list[Atoms], temperature: float, n_points_to_sample: float | Literal['all'] = 'all', sample_start_from_frame: float = 5000, compute_total_pdf: bool = False):
        self.traj = traj
        self.temperature = temperature
        self.n_atoms = len(traj[0])
        self.n_frames = len(traj)
        self.n_points_to_sample = n_points_to_sample
        self.sample_start_from_frame = sample_start_from_frame
        self.compute_total_pdf = compute_total_pdf
        
    def __call__(self):
        velocity_kdes, velocities_by_symbol = self.sample_velocity_distributions()
        max_velocity = max(max(velocities) for velocities in velocities_by_symbol.values())
        v_values, mb_total_pdf, pdf_by_element = self.maxwell_boltzmann_distribution(v_range=(0, max_velocity))
        
        empirical = {k: velocity_kdes[k](v_values) for k in velocity_kdes.keys()}
        if self.compute_total_pdf:
            all_velocities = np.array([v for vels in velocities_by_symbol.values() for v in vels])
            overall_kde = stats.gaussian_kde(all_velocities)
            empirical['total'] = overall_kde(v_values)
        
        mb = {k: pdf_by_element[k] for k in pdf_by_element.keys()}
        if self.compute_total_pdf:
            mb['total'] = mb_total_pdf
        
        return {'empirical': empirical, 'maxwell_boltzmann': mb}
        
    def sample_velocity_distributions(self):
        trajectory = self.traj
        start_frame = self.sample_start_from_frame
        n_points = self.n_points_to_sample
        
        if len(trajectory) - start_frame <= 0:
            raise ValueError(f"start_frame {start_frame} is beyond the trajectory length {len(trajectory)}")
        
        velocities_by_symbol = {k: None for k in np.unique(trajectory[0].get_chemical_symbols())}
        atom2idx = {
            symbol: np.array([i for i, s in enumerate(trajectory[0].get_chemical_symbols()) if s == symbol]) 
            for symbol in velocities_by_symbol.keys()
        }
        
        for symbol in velocities_by_symbol.keys():
            atom_indices = atom2idx[symbol]
            gathered_velocities = np.concatenate(
                [np.linalg.norm(trajectory[i].get_velocities()[atom_indices], axis=1) for i in range(start_frame, len(trajectory))],
                axis=0
            )
            if isinstance(n_points, str) and n_points == 'all':
                velocities_by_symbol[symbol] = gathered_velocities
            else:
                random_indices = np.random.choice(gathered_velocities.shape[0], size=n_points, replace=False)
                velocities_by_symbol[symbol] = gathered_velocities[random_indices]
        
        velocity_kdes = {symbol: stats.gaussian_kde(velocities) for symbol, velocities in velocities_by_symbol.items()}
        return velocity_kdes, velocities_by_symbol
    
    def maxwell_boltzmann_distribution(self, v_range: tuple[float, float] | None=None, return_total_pdf: bool=False):
        atoms = self.traj[0]
        temperature = self.temperature
        n_points = self.n_points_to_sample if not isinstance(self.n_points_to_sample, str) else 1000
        
        masses = atoms.get_masses()    
        unique_masses = {atom.symbol: masses[i] for i, atom in enumerate(atoms)}
        
        kb = units.kB
        if v_range is None:
            min_mass = min(masses)
            v_mp = np.sqrt(2 * kb * temperature / min_mass)
            v_range = (0, 5 * v_mp)
            
        v_values = np.linspace(v_range[0], v_range[1], n_points)
        
        total_pdf = np.zeros_like(v_values) if return_total_pdf else None
        pdf_by_element = {}
        
        for symbol, mass in unique_masses.items():
            prefactor = 4 * np.pi * (mass / (2 * np.pi * kb * temperature))**(3/2)
            exponent = -mass * v_values**2 / (2 * kb * temperature)
            pdf = prefactor * v_values**2 * np.exp(exponent)
            pdf_by_element[symbol] = pdf
            
            if return_total_pdf:
                weight = sum(1 for atom in atoms if atom.symbol == symbol) / len(atoms)
                total_pdf += pdf * weight
        
        return v_values, total_pdf, pdf_by_element

def diffusivity_coefficient(traj: list[Atoms], figname: str, timestep: float = 1.0):
    positions = np.array([atoms.get_positions() for atoms in traj])
    cell = traj[0].get_cell()
    inv_cell = np.linalg.inv(cell.T)

    unwrapped_positions = positions.copy()
    for i in range(1, len(traj)):
        disp = positions[i] - positions[i - 1]
        frac_disp = np.dot(disp, inv_cell)
        frac_disp -= np.round(frac_disp)
        cart_disp = np.dot(frac_disp, cell.T)
        unwrapped_positions[i] = unwrapped_positions[i - 1] + cart_disp
        
    n_steps = len(traj)
    msd = np.zeros(n_steps)
    for i in range(n_steps):
        displacements = unwrapped_positions[i:] - unwrapped_positions[:-i or None]
        sq_disp = np.sum(displacements**2, axis=2)
        msd[i] = np.mean(sq_disp)
    
    plt.figure()
    plt.plot(msd)
    plt.savefig(figname)
    plt.close()
    
    start = int(n_steps * 0.2)
    end = int(n_steps * 0.8)
    time = np.arange(n_steps) * timestep
    slope, _, _, _, _ = linregress(time[start:end], msd[start:end])
    diffcoef = slope / 6  # Diffusion coefficient in Å^2/ps
    return diffcoef

def compute_multipliers(interface: Structure, film: Structure, subs: Structure):
    set_film = film.chemical_system_set
    set_subs = subs.chemical_system_set
    elem_interface = {x: len(interface.indices_from_symbol(x)) for x in interface.chemical_system_set}
    elem_film = {x: len(film.indices_from_symbol(x)) for x in set_film}
    elem_subs = {x: len(subs.indices_from_symbol(x)) for x in set_subs}

    if set(set_film).issubset(set_subs):
        set1 = set_subs.copy()
        set2 = set_film.copy()
        for x in set2: set1.remove(x)
        pick_unique_elem = list(set1)[0]
        subs_multiplier = elem_interface[pick_unique_elem] / elem_subs[pick_unique_elem]
        pick_common_elem = list(set2)[0]
        film_multiplier = (elem_interface[pick_common_elem] - elem_subs[pick_common_elem] * subs_multiplier) / elem_film[pick_common_elem]
    elif set(set_subs).issubset(set_film):
        set1 = set_film.copy()
        set2 = set_subs.copy()
        for x in set2: set1.remove(x)
        pick_unique_elem = list(set1)[0]
        film_multiplier = elem_interface[pick_unique_elem] / elem_film[pick_unique_elem]
        pick_common_elem = list(set2)[0]
        subs_multiplier = (elem_interface[pick_common_elem] - elem_film[pick_common_elem] * film_multiplier) / elem_subs[pick_common_elem]
    else:
        set1 = set_film.copy()
        set2 = set_subs.copy()
        mult = []
        for x in set1: mult.append(elem_interface[x] / elem_film[x])
        film_multiplier = min(mult)
        mult = []
        for x in set2: mult.append(elem_interface[x] / elem_subs[x])
        subs_multiplier = min(mult)
    return film_multiplier, subs_multiplier

def compute_interfacial_energy(energy_interface, energy_film, energy_subs, film_multiplier, subs_multiplier):
    return energy_interface - film_multiplier * energy_film - subs_multiplier * energy_subs

def compute_formation_energy_for_interfaces(energy_interface, energy_film_bulk, energy_subs_bulk, film_multiplier, subs_multiplier, film_scaling, subs_scaling):
    return energy_interface - film_scaling * film_multiplier * energy_film_bulk - subs_scaling * subs_multiplier * energy_subs_bulk

def compute_formation_energy_for_slabs(energy_slab, energy_bulk, multiplier):
    return energy_slab - multiplier * energy_bulk

def compute_coordination_number(structure: Structure):
    cnn = CrystalNN()
    return [cnn.get_cn(structure, i) for i in range(structure.num_sites)]

def compute_rms_dist(struct1: Structure, struct2: Structure, return_max=False):
    sm = StructureMatcher(stol=1)
    dist = sm.get_rms_dist(struct1, struct2)
    return (dist[0], dist[1]) if dist and return_max else (dist[0] if dist else None, None)

def compute_gap_bw_layers(structure: Structure):
    if np.isnan(structure.frac_coords).any():
        return None, None, None

    lat_mat = structure.lattice.matrix
    basal_area = np.linalg.norm(np.cross(lat_mat[:, 0], lat_mat[:, 1]))

    if basal_area < 1e-8:
        return None, None, None

    max_height = structure.volume / basal_area

    z_layers = np.sort(np.unique(np.mod(structure.cart_coords[:, 2], max_height)))

    if len(z_layers) < 2:
        return None, None, None

    internal_gaps = np.diff(z_layers)
    pbc_gap = max_height - (z_layers.max() - z_layers.min())
    all_gaps = np.sort(np.append(internal_gaps, pbc_gap))[::-1]

    if len(all_gaps) == 0:
        return None, None, None
    elif len(all_gaps) == 1:
        return all_gaps[0], all_gaps[0], all_gaps[0]
    else:
        return all_gaps[0], all_gaps[1], all_gaps[-1]

def compute_rdf_for_trajectory(trajectory, rmax: float, dr: float = 0.5):
    distances = np.array([])
    for step in trajectory:
        structure = Structure.from_ase_atoms(step)
        distances = np.append(distances, structure.get_neighbor_list(r = rmax)[3])
    nbins = int(np.ceil(rmax / dr))
    hist, binedges = np.histogram(distances, range = (0, rmax), bins = nbins)
    rdf_x = 0.5 * (binedges[1:] + binedges[:-1]) + 0.5 * rmax / nbins
    binvolume = (4 / 3) * np.pi * (binedges[1:] ** 3 - binedges[:-1] ** 3)
    density = structure.num_sites / structure.volume
    rdf_y = hist / (binvolume * density * structure.num_sites * len(trajectory))
    return rdf_x, rdf_y

def compute_rdf_hellinger_distance(rdf1, rdf2):
    return np.linalg.norm(np.sqrt(rdf1) - np.sqrt(rdf2)) / np.sqrt(2)

def compute_rdf_kl_divergence(rdf1, rdf2):
    return kl_div(rdf1, rdf2).sum()

def compute_min_distance(traj: list[Atoms], use_half_initial: bool=True, fixed_threshold: float=None, enable_tqdm: bool=True):
    """
    Check if the minimum interatomic distance in any frame falls below a threshold.

    Mathematical computation:
        For each frame, compute the minimum value of the distance matrix (excluding self-distances).
        If min_dist <= threshold, the system is considered non-physical.
        threshold = 0.5 * min_dist_initial (if use_half_initial) or fixed_threshold.
    """
    threshold = fixed_threshold
    if threshold is None and use_half_initial:
        dist_matrix = traj[0].get_all_distances(mic=True)
        np.fill_diagonal(dist_matrix, np.inf)
        threshold = 0.5 * np.min(dist_matrix)
    
    min_distance_found = np.inf
    for k, atoms in enumerate(tqdm(traj, disable=not enable_tqdm, desc="min distance")):
        dist_matrix = atoms.get_all_distances(mic=True)
        np.fill_diagonal(dist_matrix, np.inf)        
        min_dist = np.min(dist_matrix)
        min_distance_found = min(min_distance_found, min_dist)
        
        if min_dist <= threshold:
            min_indices = np.where(dist_matrix == min_dist)
            return {'is_physical': False, 'threshold': threshold, 'first_violation': k, 'min_distance': min_distance_found, 'violation_atoms': (int(min_indices[0][0]), int(min_indices[1][0]))}
    
    return {'is_physical': True, 'threshold': threshold, 'first_violation': None, 'min_distance': min_distance_found, 'violation_atoms': None}

def compute_dynamic_knn_preservation(traj: list[Atoms], scale: float = 0.2, n_neighbors: int = 4, mic: bool = True, enable_tqdm: bool = True):
    """
    Check if the average distance to dynamic k-nearest neighbors deviates beyond a threshold.

    Mathematical computation:
        For each atom, compute avg distance to current neighbors in each frame.
        If deviation > scale * initial_avg_distance, report violation.
    """
    if len(traj) <= 1:
        return {'is_physical': True} | None

    initial_frame = traj[0]
    n_atoms = len(initial_frame)
    initial_dist_matrix = initial_frame.get_all_distances(mic=mic)
    np.fill_diagonal(initial_dist_matrix, np.inf)

    if n_neighbors >= n_atoms:
        n_neighbors = n_atoms - 1

    initial_neighbor_indices = np.argpartition(initial_dist_matrix, n_neighbors, axis=1)[:, :n_neighbors]
    initial_avg_distances = np.mean(
        np.take_along_axis(
            initial_dist_matrix, initial_neighbor_indices, axis=1
        ), axis=1
    )
    thresholds = scale * initial_avg_distances

    status_dict = {
        'frame': -1,
        'atom_index': -1,
        'initial_avg_distance': None,
        'current_avg_distance': None,
        'initial_neighbors': None,
        'current_neighbors': None,
        'deviation': 0.0,
        'threshold': None
    }

    for k in tqdm(
        range(1, len(traj)), disable=not enable_tqdm, desc="dynamic knn preservation"
    ):
        current_frame = traj[k]
        current_dist_matrix = current_frame.get_all_distances(mic=mic)
        np.fill_diagonal(current_dist_matrix, np.inf)

        current_neighbor_indices = np.argpartition(
            current_dist_matrix, n_neighbors, axis=1
        )[:, :n_neighbors]
        current_avg_distances = np.mean(
            np.take_along_axis(
                current_dist_matrix, current_neighbor_indices, axis=1
            ), axis=1
        )

        deviations = np.abs(current_avg_distances - initial_avg_distances)
        max_deviation_idx = np.argmax(deviations)
        max_deviation = deviations[max_deviation_idx]

        if max_deviation > status_dict['deviation']:
            status_dict['frame'] = k
            status_dict['atom_index'] = max_deviation_idx
            status_dict['initial_avg_distance'] = initial_avg_distances[max_deviation_idx]
            status_dict['current_avg_distance'] = current_avg_distances[max_deviation_idx]
            status_dict['initial_neighbors'] = initial_neighbor_indices[max_deviation_idx]
            status_dict['current_neighbors'] = current_neighbor_indices[max_deviation_idx]
            status_dict['deviation'] = max_deviation
            status_dict['threshold'] = thresholds[max_deviation_idx]

        violations = deviations > thresholds

        if np.any(violations):
            return {'is_physical': False} | status_dict

    return {'is_physical': True} | status_dict

def compute_large_velocity(traj: list[Atoms], temperature: float, scale: float = 5., n_violations: int = 5, among_consecutive_frames: int = 10, enable_tqdm: bool = True):
    """
    Detect atoms with velocities exceeding a scaled Maxwell-Boltzmann threshold for several consecutive frames.

    Mathematical computation:
        v_threshold = sqrt(2 * kB * T / m) * scale
        If an atom exceeds v_threshold in n_violations frames within among_consecutive_frames, report violation.
    """
    v_thresholds = np.sqrt(2 * units.kB * temperature / traj[0].get_masses()) * scale
    violations_tracking = {}
    
    for frame_idx, atoms in enumerate(tqdm(traj, desc="large velocity", disable=not enable_tqdm)):
        velocity_magnitudes = np.linalg.norm(atoms.get_velocities(), axis=1)
        for atom_idx in np.where(velocity_magnitudes > v_thresholds)[0]:
            if atom_idx not in violations_tracking:
                violations_tracking[atom_idx] = {'count': 0, 'frames': []}
            violations_tracking[atom_idx]['frames'].append(frame_idx)
            
            if len(violations_tracking[atom_idx]['frames']) >= n_violations:
                recent_frames = violations_tracking[atom_idx]['frames'][-n_violations:]
                if recent_frames[-1] - recent_frames[0] < among_consecutive_frames:
                    return {'is_physical': False, 'frame_index': frame_idx, 'atom_index': atom_idx, 'start_frame': recent_frames[0], 'end_frame': recent_frames[-1]}
    return {'is_physical': True}

def compute_lindemann_index(traj: list[Atoms], local_threshold: float = 0.1, global_threshold: float = 0.1, enable_tqdm: bool = True):
    """
    Compute the Lindemann index to assess melting or disorder.

    Mathematical computation:
        For each atom i:
            q_local[i] = (1/(N-1)) * sum_j sqrt(<r_ij^2> - <r_ij>^2) / <r_ij>
        q_global = mean(q_local)
        If any q_local > local_threshold or q_global > global_threshold, report violation.
    """
    n_frames, n_atoms = len(traj), len(traj[0])
    r_sum, r2_sum = np.zeros((n_atoms, n_atoms)), np.zeros((n_atoms, n_atoms))
    
    for atoms in tqdm(traj, disable=not enable_tqdm, desc="lindemann index setup"):
        dist = atoms.get_all_distances(mic=True)
        r_sum += dist
        r2_sum += dist**2
    
    r_avg, r2_avg = r_sum / n_frames, r2_sum / n_frames
    q_local = np.zeros(n_atoms)
    for i in range(n_atoms):
        numerator = np.sqrt(r2_avg[i, :] - r_avg[i, :]**2)
        mask = (i != np.arange(n_atoms)) & (r_avg[i, :] > 0)
        if np.sum(mask) > 0:
            q_local[i] = (1 / (n_atoms - 1)) * np.sum(numerator[mask] / r_avg[i, :][mask])
    
    q_global = np.mean(q_local)
    return {'is_physical': not (any(q_local > local_threshold) or q_global > global_threshold), 'q_global': q_global, 'max_q_local': np.max(q_local)}
    
def compute_velocity_distribution_divergence(traj: list[Atoms], threshold: float, temperature: float, n_points: int | Literal['all'] = 'all', start_at_frame: int = 5000):
    mb_computer = MaxwellBoltzmannComputer(traj, temperature, n_points_to_sample=n_points, sample_start_from_frame=start_at_frame)
    pdfs = mb_computer()
    
    hellinger_distances = {}
    for symbol in pdfs['empirical'].keys():
        if symbol in pdfs['maxwell_boltzmann']:
            p, q = np.asarray(pdfs['empirical'][symbol]), np.asarray(pdfs['maxwell_boltzmann'][symbol])
            p, q = p / np.sum(p), q / np.sum(q)
            hellinger_distances[symbol] = np.sqrt(1 - np.sum(np.sqrt(p * q)))
            
    max_dist = max(hellinger_distances.values()) if hellinger_distances else 0
    return {'is_physical': max_dist <= threshold, 'hellinger_distances': hellinger_distances}

def compute_velocity_skewness(traj: list[Atoms], threshold: float, n_points: int | Literal['all'] = 'all', sample_start_from_frame: int = 5000):
    """
    Check if the skewness of the velocity distribution for each atom type exceeds a threshold.

    Mathematical computation:
        For each atom type, compute skewness of the velocity KDE.
        If abs(skewness) > threshold, report violation.
    """
    computer = MaxwellBoltzmannComputer(
        traj,
        n_points_to_sample=n_points,
        sample_start_from_frame=sample_start_from_frame,
        temperature=0,
    )
    velocity_kdes, velocities_by_symbol = computer.sample_velocity_distributions()

    skewness_by_symbol = {}
    for symbol in velocities_by_symbol.keys():
        velocities = velocities_by_symbol[symbol]
        v_range = np.linspace(0, np.max(velocities), n_points)
        skewness_by_symbol[symbol] = skew(velocity_kdes[symbol](v_range))

    is_physical = all(
        abs(skewness) < threshold for skewness in skewness_by_symbol.values()
    )
    return {'is_physical': is_physical, 'skewness': skewness_by_symbol}
