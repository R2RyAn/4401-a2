from sympy.codegen.ast import Raise

from .probability_agent import ProbabilityAgent
from utils import vector_to_direction
from probability import EchoGrid, DistributionModel
from state import *
from collections import Counter
import inspect


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

        # Update the echo grid with the current state
        self._echo_grid.update(state)

        # Call the echo grid to get the echo distribution
        echo_dist = self._echo_grid.get_echo_distribution()


        # if echo_dist is all 0, there is no new information, meaning we must reset our thoughts
        if all(val == 0 for val in echo_dist.values()):
            self.reset_thoughts()
            return

        # Create a new Couter for the new thoughts which will be a multiplication of the echo distribution and the current thoughts (Bayesian)
        thoughts_2 = Counter()
        # Make sure we are going over all valid position
        for value in self._valid_positions:
            thoughts_2[value] = self._thoughts[value] * echo_dist[value]

        # Sum up the new values in thought_two
        total = sum(thoughts_2.values())

        # Make sure the total is larger than 0, if it is then normalize the distribution, if it isn't then reset the thoughts
        if total > 0:
            DistributionModel.normalize(thoughts_2)
            self._thoughts = thoughts_2
        else:
            self.reset_thoughts()


    def predict(self, state):
        # We have to first pick one mouse to chase
        mice = state.get_mouse_locations()

        # Make sure there is a mouse on the board
        if mice:
            # Pick the first mouse
            self._target_mouse = mice[0]
        else:
            # If not mouse on the board then return and end program
            self.reset_thoughts()
            return

        # Create a new counter for the new thoughts
        thought_2 = Counter()

        # Loop over only valid positions
        for pos_new in self._valid_positions:
            # Initialize probability for the new position
            prob = 0
            # Loop over all valid current positions
            for curr_pos in self._valid_positions:
                # If the current position has 0 or not probability then skip it
                if self._thoughts[curr_pos] <= 0:
                    continue

                # Get the movement probability distribution from the current position
                mvmt_dist = DistributionModel.get_movement_distribution(state, curr_pos)


                # Update the probability of the new position based on the current poistion, and where it could have come from
                prob += self._thoughts[curr_pos] * mvmt_dist.get(pos_new, 0)

            # Store the updated probability for the new position in the new counter
            thought_2[pos_new] = prob

        # Normalize the new thoughts distribution and make sure it is not all 0
        if sum(thought_2.values()) > 0:
            DistributionModel.normalize(thought_2)
        else:
            self.reset_thoughts()
            thought_2 = self._thoughts.copy()

        # update the thoughts with the new thoughts
        self._echo_grid.update(state)
        echo_dist = self._echo_grid.get_echo_distribution()

        # Make sure you only keep track of the echo distribution at the targeted mouse's position
        echo_2 = Counter()
        for pos in self._valid_positions:
            # Only keep echo evidence at the target mouse's position; zero elsewhere.
            if pos == self._target_mouse:
                echo_2[pos] = echo_dist[pos]
            else:
                echo_2[pos] = 0

        # Combining the new thoughts with the filtered echo distribution
        thought_3 = Counter()
        # loop over all valid positions and combine the new thoughts with the echo distribution
        for pos in self._valid_positions:
            thought_3[pos] = thought_2[pos] + echo_2[pos]

        # Normalize the final thoughts distribution
        if sum(thought_3.values()) > 0:
            DistributionModel.normalize(thought_3)
        else:
            thought_3 = thought_2.copy()

        self._thoughts = thought_3



