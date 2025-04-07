import math
from collections import Counter
from .distribution_model import DistributionModel


class ParticleGrid:

    def __init__(self, valid_positions, particle_count=400):
        self._particle_count = particle_count
        self._valid_positions = valid_positions
        self.reset()

    def reset(self):
        # Question 3, your reset implementation goes here.
        # Recall this method resets the particle distribution to be a uniform distribution.
        # Make sure to have the particle distribution be a _Counter_ not a regular dictionary!

        # Initialize the particle distribution Counter
        self._particle_distribution = Counter()

        # Calculate how many particles should be placed at each valid position
        particles_per_position = math.floor(self._particle_count / len(self._valid_positions))

        # Distribute particles evenly across all valid positions
        for position in self._valid_positions:
            self._particle_distribution[position] = particles_per_position



    def reweight_particles(self, distribution):
        # Qustion 4, your reweight particles implementation goes here.
        # This method focuses on updating the particle distribution by sampling the given distribution.
        # Remember to normalize the distribution!

        # For sampling use DistributionModel.sample_distribution(distribution, sample_amount) which will
        # return a list of legal positions got by sampling the given distribution.

        #Sample from the given distribution
        sampled_positions = DistributionModel.sample_distribution(distribution, self._particle_count)

        #Count how many times each position was sampled
        self._particle_distribution = Counter(sampled_positions)

        #Normalize the counter so it becomes a probability distribution
        DistributionModel.normalize(self._particle_distribution)

        # Ensure all valid positions have entries in the counter, even if their count is 0
        for pos in self._valid_positions:
            if pos not in self._particle_distribution:
                self._particle_distribution[pos] = 0.0


    def get_particle_distribution(self):
        return self._particle_distribution
