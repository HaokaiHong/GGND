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

temperature = 1200
xyz_file_path = f"datasets/3BPA/test_{temperature}K.xyz"
output_path = f'real_MD/{temperature}K'
checkpoint_path = './checkpoints/MACE_run-123_stagetwo.model'
molecules = read(xyz_file_path, format='extxyz', index=':')
molecule = molecules[0]

from mace.calculators import MACECalculator
calculator = MACECalculator(model_path=checkpoint_path, device='cuda')
molecule.set_calculator(calculator)

# Simulation parameters
timestep = 1.0 * fs  # Timestep in femtoseconds
friction = 0.01 / fs  # Friction coefficient in fs⁻¹
total_steps = 100000  # 100 ps = 100,000 fs / 1.0 fs per step
log_interval = 100  # Log every 1000 steps (1 ps)

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
dyn = NoseHooverChainNVT(molecule, timestep=1.0 * units.fs, temperature_K=300.0, tdamp=100.0 * units.fs, logfile=f'{output_path}/md_nh.log', trajectory=f'{output_path}/md_nh.traj')

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
print("Trajectory saved to 'md_langevin.traj'.")