from ase.io import read
from ase import Atoms
from ase.md.langevin import Langevin
from ase.calculators.emt import EMT
from ase.units import fs, kB
import numpy as np
from ase.io import Trajectory
from tqdm import tqdm
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import ase.units as units
from mace.calculators import MACECalculator
from collections import defaultdict

# Unified parameters
xyz_file_path = "datasets/3BPA/test_600K.xyz"
output_path = 'real_MD/600K'
checkpoint_path = './checkpoints/MACE_run-123_stagetwo.model'
traj_file = f'{output_path}/md_nh.traj'
bond_threshold = 1.8
tolerance = 0.5
timestep_fs = 1.0

# Read molecules
molecules = read(xyz_file_path, format='extxyz', index=':')
molecule = molecules[0]

# Set up calculator
calculator = MACECalculator(model_path=checkpoint_path, device='cuda')
molecule.set_calculator(calculator)

# Simulation parameters
temperature = 300  # Temperature in Kelvin
timestep = timestep_fs * fs  # Timestep in femtoseconds
friction = 0.01 / fs  # Friction coefficient in fs⁻¹
total_steps = 100000  # 100 ps = 100,000 fs / 1.0 fs per step
log_interval = 100  # Log every 100 steps

# Initialize output file for energies and temperature
output_file = f'{output_path}/md_energies.txt'
with open(output_file, 'w') as f:
    f.write('# Step\tTime(ps)\tPotential_E(eV)\tKinetic_E(eV)\tTotal_E(eV)\tTemperature(K)\n')

# Function to log energies and temperature
def log_energies(atoms, step, time_ps):
    pot_energy = atoms.get_potential_energy()
    kin_energy = atoms.get_kinetic_energy()
    total_energy = pot_energy + kin_energy
    temp = kin_energy / (1.5 * kB * len(atoms))  # Temperature from kinetic energy
    with open(output_file, 'a') as f:
        f.write(f'{step}\t{time_ps:.3f}\t{pot_energy:.6f}\t{kin_energy:.6f}\t{total_energy:.6f}\t{temp:.2f}\n')

MaxwellBoltzmannDistribution(molecule, temperature_K=300.0)
dyn = NoseHooverChainNVT(molecule, timestep=1.0 * units.fs, temperature_K=300.0, tdamp=100.0 * units.fs, logfile=f'{output_path}/md_nh.log', trajectory=traj_file)

# Attach the logging function
def print_energies():
    step = dyn.get_number_of_steps()
    time_ps = step * timestep / fs / log_interval  # Convert to picoseconds
    log_energies(molecule, step, time_ps)

dyn.attach(print_energies, interval=log_interval)

# Run the simulation
for _ in tqdm(range(total_steps)):
    dyn.run(1)

print("MD simulation completed. Energies and temperature saved to 'md_energies.txt'.")
print(f"Trajectory saved to '{traj_file}'.")

# Analysis functions
def analyze_bond_lengths(ref_file, bond_threshold):
    # Read all molecules from the reference file
    molecules = read(ref_file, index=':', format='extxyz')
    
    # Dictionary to store bond lengths by bond type
    bond_lengths = defaultdict(list)
    # List to store edge indices (atom index pairs for bonds)
    edge_indices = []
    
    bond_index = {}
    for mol_idx, mol in enumerate(molecules):
        # Get atomic symbols and positions
        symbols = mol.get_chemical_symbols()
        positions = mol.get_positions()
        
        # Analyze bonds
        for i in range(len(mol)):
            for j in range(i + 1, len(mol)):
                # Calculate distance between atoms i and j
                dist = np.linalg.norm(positions[i] - positions[j])
                
                # Check if distance is within bonding threshold
                if dist < bond_threshold:
                    # Store edge index: (molecule_index, atom_i, atom_j)
                    edge_indices.append((mol_idx, i, j))
                    # Store bond length with bond type
                    bond_type = f"{i}-{j}"
                    bond_lengths[bond_type].append(dist)
    
    # Calculate average bond lengths for consistent bonds
    mol_len = len(molecules)
    true_bonds = {}
    true_lengths = []
    edge_index = []
    for bond_type, lengths in bond_lengths.items():
        if len(lengths) == mol_len:
            a, b = map(int, bond_type.split('-'))
            true_bonds[bond_type] = lengths
            edge_index.append([a, b])
            true_lengths.append(np.mean(lengths))
    
    return np.array(edge_index), true_lengths

def check_edge_lengths(reference_lengths, calculated_lengths, tolerance):
    if len(reference_lengths) != len(calculated_lengths):
        raise ValueError("Reference and calculated lengths lists must have the same length")
    
    for ref, calc in zip(reference_lengths, calculated_lengths):
        if not (ref - tolerance <= calc <= ref + tolerance):
            return False
    return True

def analyze_trajectory(traj_file, edge_index, true_lengths, tolerance, timestep_fs):
    # Read trajectory
    traj = read(traj_file, index=':')
    
    stable_frames = len(traj)
    for k, atoms in enumerate(tqdm(traj)):
        pos = atoms.get_positions()
        p1 = pos[edge_index[:, 0]]  # Start points
        p2 = pos[edge_index[:, 1]]  # End points
        edges = np.sqrt(np.sum((p2 - p1) ** 2, axis=1))
        result = check_edge_lengths(true_lengths, edges, tolerance)
        if not result:
            stable_frames = k
            break
    
    # Calculate stable time
    stable_time_ps = (stable_frames - 1) * timestep_fs / 1000.0 if stable_frames > 0 else 0.0
    stable_time_fs = (stable_frames - 1) * timestep_fs if stable_frames > 0 else 0.0
    
    return stable_time_ps, stable_time_fs

# Perform analysis
edge_index, true_lengths = analyze_bond_lengths(xyz_file_path, bond_threshold)
stable_time_ps, stable_time_fs = analyze_trajectory(traj_file, edge_index, true_lengths, tolerance, timestep_fs)

print(f"Stability of the trajectory: {stable_time_ps} ps")
print(f"Stability of the trajectory: {stable_time_fs} fs")