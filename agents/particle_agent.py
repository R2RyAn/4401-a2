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
    # Update the echo grid with current state
    self._echo_grid.update(state)
    
    # Get the echo distribution
    echo_dist = self._echo_grid.get_echo_distribution()
    
    # Check if all information is new (all probabilities are 0)
    all_new = all(prob == 0 for prob in echo_dist.values())
    if all_new:
        self.reset_thoughts()
        self._particle_grid.reset()
        return
    
    # Create a Counter for the new thoughts
    new_thoughts = Counter()
    
    # Combine echo distribution with particle distribution
    for pos in self._valid_positions:
        # Multiply the echo probability with particle probability
        new_thoughts[pos] = echo_dist[pos] * self._thoughts[pos]
    
    # Normalize the distribution
    DistributionModel.normalize(new_thoughts)
    
    # Update the agent's thoughts
    self._thoughts = dict(new_thoughts)
    
    # Reweight particles based on new thoughts
    self._particle_grid.reweight_particles(self._thoughts)
    
  def predict(self, state):
    # Update the echo grid with current state
    self._echo_grid.update(state)
    
    # Create a copy of the state to avoid affecting the actual state
    state_copy = state.copy()
    
    # Create a Counter for the new thoughts
    new_thoughts = Counter()
    
    # For each valid position, simulate mouse movement
    for pos in self._valid_positions:
        # Get movement distribution for this position
        movement_dist = DistributionModel.get_movement_distribution(state_copy, pos)
        
        # For each possible new position
        for new_pos, move_prob in movement_dist.items():
            if new_pos in self._valid_positions:
                # Multiply the movement probability with current thought probability
                new_thoughts[new_pos] += self._thoughts[pos] * move_prob
    
    # Normalize the distribution
    DistributionModel.normalize(new_thoughts)
    
    # Update the agent's thoughts
    self._thoughts = dict(new_thoughts)
    
    # Reweight particles based on new thoughts
    self._particle_grid.reweight_particles(self._thoughts)
