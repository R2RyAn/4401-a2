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

        # Compute how many particles should be placed at each valid position
        particles_per_position = self._particle_count // len(self._valid_positions)

        # Distribute particles evenly across all valid positions
        for position in self._valid_positions:
            self._particle_distribution[position] = particles_per_position

        # Optional: Print the normalized distribution in the test case format
        # formatted = [
        #     [[pos[0], pos[1]], self._particle_distribution[pos] / self._particle_count]
        #     for pos in sorted(self._valid_positions)
        # ]
        # print(f'"solution Normal": {formatted}')

    def reweight_particles(self, distribution):
        # Qustion 4, your reweight particles implementation goes here.
        # This method focuses on updating the particle distribution by sampling the given distribution.
        # Remember to normalize the distribution!

        # For sampling use DistributionModel.sample_distribution(distribution, sample_amount) which will
        # return a list of legal positions got by sampling the given distribution.

        # Step 1: Sample from the given distribution
        sampled_positions = DistributionModel.sample_distribution(distribution, self._particle_count)

        # Step 2: Count how many times each position was sampled (i.e. build new particle distribution)
        self._particle_distribution = Counter(sampled_positions)

        # Step 3: Normalize the counter so it becomes a probability distribution
        DistributionModel.normalize(self._particle_distribution)

        # Print the normalized distribution (no need to divide by particle_count again)
        formatted = [
            [[pos[0], pos[1]], self._particle_distribution[pos]]
            for pos in sorted(self._valid_positions, key=lambda x: (x[0], x[1]))
        ]
        print(f'"solution Weighted": {formatted}')

    def get_particle_distribution(self):
        return self._particle_distribution
