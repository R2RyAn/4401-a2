from .probability_agent import ProbabilityAgent
from utils import vector_to_direction
from probability import *
from state import *
from collections import Counter


class ParticleAgent(ProbabilityAgent):

    def __init__(self, valid_positions):
        super().__init__(valid_positions)
        self._particle_grid = ParticleGrid(self._valid_positions)
        self._echo_grid = EchoGrid()
        self._thoughts = self._particle_grid.get_particle_distribution()

    # Helpful Hints and Functions:
    # EchoGrid.get_echo_distribution() --> returns a distribution over all legal positions on the map as a dictionary
    #                                      where the key is a position and value is the probability of a mouse being there.
    # ProbabilityAgent.reset_thoughts() --> resets self._thoughts to be uniform (i.e. agent thinks all positions may have a mouse)
    # DistributionModel.normalize(distribution) --> normalizes the given distribution
    # DistributionModel.get_movement_distribution(state, agent_pos) --> returns a movement distribution for the given agent through it's position
    # GameState.copy() --> returns a copy
    # GameStateHandler.move_mouse(old_pos, new_pos) --> moves the mouse from the old position to the new position on the map
    # ParticleGrid.reset() --> Resets the particle distribution to be uniform
    # ParticleGrid.reweight_particles(distribution) --> reweights the particles based on the given distribution

    # Instead of using a regular dictionary we recommend you use a Counter object to avoid needing to check for keys before using
    # them. Counters default any unseen key to the value of 0.

    # Remember to normalize brefore updating the agents thoughts and to look over only valid positions (use self._valid_positions).

    def listen(self, state):
        # Question 5, your ParticleAgent listen solution goes here.
        # Similar to the MarkovAgent this method uses echo distributions from the EchoGrid but it also uses particle distributions
        # provided by the ParticleGrid build the agent's thoughts. Just like the MarkovAgent there is a special case to consider which
        # happens when the distribution given by the EchoGrid has only information which has NOT been seen before. In this case you must
        # reset your current thought distribution AND the particle distribution before continuing. Remember to reweight the particles after
        # updating the agent's thoughts.

        # Uncomment this when you start implementing
        self._echo_grid.update(state)
        
        # Get the echo distribution
        echo_distribution = self._echo_grid.get_echo_distribution()
        
        # Check if we have any overlapping positions between echo and thoughts
        overlap = False
        for pos in echo_distribution:
            if pos in self._thoughts and self._thoughts[pos] > 0:
                overlap = True
                break
        
        # Special case: If no overlap, reset thoughts and particles
        if not overlap:
            self.reset_thoughts()
            self._particle_grid.reset()
            self._thoughts = self._particle_grid.get_particle_distribution()
            return
        
        # Update thoughts by multiplying echo distribution with current thoughts
        updated_thoughts = Counter()
        for pos in self._valid_positions:
            updated_thoughts[pos] = self._thoughts[pos] * echo_distribution.get(pos, 0)
        
        # Normalize the updated thoughts
        DistributionModel.normalize(updated_thoughts)
        
        # Update the agent's thoughts
        self._thoughts = updated_thoughts
        
        # Reweight particles based on the updated thoughts
        self._particle_grid.reweight_particles(self._thoughts)


    def predict(self, state):
        # Question 6, your ParticleAgent predict solution goes here.
        # Recall for the predict method we want to track one mouse down at a time by "predicting their moves". This should
        # be done through moving the mouse into positions on the map using this distribution to update your thoughts.
        # To avoid annoyances of state manipulation you should use a copy of the given state when you pretend to move the mouse
        # so that it does not effect the actual state.

        # As an addition the special case from above carries through to this method and should be handled in the same fashion
        # and remember to reweight the particles after updating the agent's thoughts.

        # Update the echo grid with current state
        self._echo_grid.update(state)  # Do Not Remove, it is required to have the EchoGrid give accurate information
        
        # Get echo distribution
        echo_distribution = self._echo_grid.get_echo_distribution()
        
        # Handle special case: check for overlap between echo and thoughts
        overlap = False
        for position in self._valid_positions:
            if position in echo_distribution and position in self._thoughts:
                if echo_distribution[position] > 0 and self._thoughts[position] > 0:
                    overlap = True
                    break
        
        # If no overlap, reset thoughts and particles
        if not overlap:
            self.reset_thoughts()
            self._particle_grid.reset()
            self._thoughts = self._particle_grid.get_particle_distribution()
            return

        # Create a new distribution for new thoughts
        updated_thoughts = Counter()
        
        # For each valid current position
        for prev_pos in self._valid_positions:
            if self._thoughts[prev_pos] <= 0:
                continue
                

            # Make a copy to avoid modifying actual state
            state_copy = state.copy()
            # Get the movement distribution for this position
            movement_dist = DistributionModel.get_movement_distribution(state_copy, prev_pos)
            
            # Distribute thoughts according to movement probabilities
            for new_pos, move_prob in movement_dist.items():
                updated_thoughts[new_pos] += self._thoughts[prev_pos] * move_prob
        
        # Normalize the updated distribution
        if sum(updated_thoughts.values()) > 0:
            DistributionModel.normalize(updated_thoughts)
        else:
            # If all probabilities are zero, reset
            self.reset_thoughts()
            self._particle_grid.reset()
            return
        
        # Update the agent's thoughts
        self._thoughts = updated_thoughts
        
        # Reweight particles based on updated thoughts
        self._particle_grid.reweight_particles(self._thoughts)
        

