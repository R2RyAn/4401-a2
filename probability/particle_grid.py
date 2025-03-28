from collections import Counter
from .distribution_model import DistributionModel

class ParticleGrid:

  def __init__(self, valid_positions, particle_count=400):
    self._particle_count = particle_count
    self._valid_positions = valid_positions
    self.reset()

  def reset(self):
    # Create a Counter for uniform distribution
    self._particle_distribution = Counter()
    
    # Calculate particles per position (will be evenly divisible as per requirements)
    particles_per_position = self._particle_count // len(self._valid_positions)
    
    # Distribute particles evenly across all valid positions
    for pos in self._valid_positions:
        self._particle_distribution[pos] = particles_per_position

  def reweight_particles(self, distribution):
    # Sample positions based on the given distribution
    sampled_positions = DistributionModel.sample_distribution(distribution, self._particle_count)
    
    # Create new Counter with sampled positions
    self._particle_distribution = Counter(sampled_positions)
    
    # Normalize the distribution
    DistributionModel.normalize(self._particle_distribution)

  def get_particle_distribution(self):
    return self._particle_distribution
