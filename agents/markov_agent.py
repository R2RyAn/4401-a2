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
        flag = True

        # Step 1: Update the echo grid with the current state (get new mouse sounds)
        self._echo_grid.update(state)

        # Step 2: Get the echo distribution (where the sounds are coming from)
        echo_dist = self._echo_grid.get_echo_distribution()

        # Step 3: If echo contains no useful info, reset thoughts to uniform

        if all(val == 0 for val in echo_dist.values()):
            self.reset_thoughts()
            return

        # Step 4: Create a new belief map by combining echo info with current beliefs Bayesian Style
        thoughts_two = Counter()
        for value in self._valid_positions:
            thoughts_two[value] = self._thoughts[value] * echo_dist[value]

        # Step 5: Normalize and update the agent's thoughts
        total = sum(thoughts_two.values())
        if total > 0:
            DistributionModel.normalize(thoughts_two)
            self._thoughts = thoughts_two
        else:
            self.reset_thoughts()
        print("\n==================")
        print(f"[DEBUG] Mice: {state.get_mouse_locations()}")
        print("[🧠 Your Thoughts]")
        for pos in sorted(self._thoughts):
            print(f"{pos}: {self._thoughts[pos]:.16f}")
        print("==================\n")

    def predict(self, state):
        # --- Step A: Choose a target mouse if not set or lost ---
        mice = state.get_mouse_locations()
        if mice:
            self._target_mouse = mice[0]  # choose the first mouse (or choose by some criteria)
        else:
            # No mice: reset thoughts and return
            self.reset_thoughts()
            return

        # --- Step 1: Time Elapse - Predict mouse movement using movement model ---
        new_thoughts = Counter()
        for new_pos in self._valid_positions:
            prob_at_new_pos = 0
            for curr_pos in self._valid_positions:
                if self._thoughts[curr_pos] <= 0:
                    continue
                movement_dist = DistributionModel.get_movement_distribution(state, curr_pos)
                prob_at_new_pos += self._thoughts[curr_pos] * movement_dist.get(new_pos, 0)
            new_thoughts[new_pos] = prob_at_new_pos

        if sum(new_thoughts.values()) > 0:
            DistributionModel.normalize(new_thoughts)
        else:
            self.reset_thoughts()
            new_thoughts = self._thoughts.copy()

        # --- Step 2: Update echo grid and get echo distribution ---
        self._echo_grid.update(state)
        echo_dist = self._echo_grid.get_echo_distribution()

        # --- Step 3: Filter echo distribution to focus only on the target mouse ---
        filtered_echo = Counter()
        for pos in self._valid_positions:
            # Only keep echo evidence at the target mouse's position; zero elsewhere.
            if pos == self._target_mouse:
                filtered_echo[pos] = echo_dist[pos]
            else:
                filtered_echo[pos] = 0

        # --- Step 4: Combine time-elapsed beliefs and echo evidence via weighted averaging ---
        final_thoughts = Counter()
        prior_weight = 0.4  # weight for time-elapsed beliefs
        evidence_weight = 0.6  # weight for echo evidence
        for pos in self._valid_positions:
            final_thoughts[pos] = (new_thoughts[pos]) + (filtered_echo[pos])

        if sum(final_thoughts.values()) > 0:
            DistributionModel.normalize(final_thoughts)
        else:
            final_thoughts = new_thoughts.copy()

        self._thoughts = final_thoughts

        # --- Debug output ---
        print("\n==================")
        print(f"[DEBUG] Mice: {state.get_mouse_locations()}")
        print(f"[DEBUG] Chasing mouse: {self._target_mouse}")
        print("[🧠 Your Thoughts]")
        for pos in sorted(self._thoughts):
            print(f"{pos}: {self._thoughts[pos]:.16f}")
        print("==================\n")

