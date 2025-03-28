from .probability_agent import ProbabilityAgent
from utils import vector_to_direction
from probability import EchoGrid, DistributionModel
from state import *
from collections import Counter


class MarkovAgent(ProbabilityAgent):

    def __init__(self, valid_positions):
        super().__init__(valid_positions)
        self._echo_grid = EchoGrid()

    # Helpful Hints and Functions:
    # EchoGrid.get_echo_distribution() --> returns a distribution over all legal positions on the map as a dictionary
    #                                      where the key is a position and value is the probability of a mouse being there.
    # ProbabilityAgent.reset_thoughts() --> resets self._thoughts to be uniform (i.e. agent thinks all positions may have a mouse)
    # DistributionModel.normalize(distribution) --> normalizes the given distribution
    # DistributionModel.get_movement_distribution(state, agent_pos) --> returns a movement distribution for the given agent through it's position.
    # GameState.copy() --> returns a copy
    # GameStateHandler.move_mouse(old_pos, new_pos) --> moves the mouse from the old position to the new position on the map

    # Instead of using a regular dictionary we recommend you use a Counter object to avoid needing to check for keys before using
    # them. Counters default any unseen key to the value of 0.

    # Remember to normalize before updating the agents thoughts and to look over only valid positions (use self._valid_positions).

    def listen(self, state):
        flag = True
        # Step 1: Update the echo grid with the current state (get new mouse sounds)
        self._echo_grid.update(state)

        # Step 2: Get the echo distribution (where the sounds are coming from)
        echo_dist = self._echo_grid.get_echo_distribution()

        # Step 3: If echo contains no useful info, reset thoughts to uniform
        for val in echo_dist.values():
            if val > 0:
                flag = False
                break
            if flag:
                self.reset_thoughts()
                return

        # Step 4: Create a new belief map by combining echo info with current beliefs Bayesian Style
        thoughts_two = Counter()
        for value in self._valid_positions:
            thoughts_two[value] = self._thoughts[value] * echo_dist[value]

        # Step 5: Normalize and update the agent's thoughts
        if sum(thoughts_two.values()) > 0:
            DistributionModel.normalize(thoughts_two)
            self._thoughts = thoughts_two
        else:
            self.reset_thoughts()

    # Implement the Time Lapse for HMM (Question 3)
    def predict(self, state):
        # Update the echo grid with current state
        self._echo_grid.update(state)


