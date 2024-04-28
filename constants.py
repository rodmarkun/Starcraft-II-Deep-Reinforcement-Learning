import numpy as np
from gymnasium import spaces

# Define the number of discrete actions available for the DRL agent to take.
# These actions are:
#   0 - Expand, Build Assimilators and Workers: Focuses on long-term economic development.
#   1 - Build Stargate: Constructs a military building essential for creating air units.
#   2 - Build Void Rays: Produces Void Rays, a powerful military air unit.
#   3 - Attack with all Void Rays: Commands all available Void Rays to attack the enemy, potentially gaining rewards.
#   4 - Build Pylon: Constructs Pylons, which are necessary for supplying and powering buildings and units.
#   5 - Do nothing: Takes no action during this step, which can be strategic in certain scenarios.
NUMBER_OF_ACTIONS = 6

# Define the observation space of the environment that the DRL agent interacts with.
# The elements in the observation space are:
#   ProbeNum: The number of Probes (worker units).
#   VRNum: The number of Void Rays (military air units).
#   AttackingVRs: The number of Void Rays currently engaged in combat.
#   NexusNum: The number of Nexuses (main base buildings).
#   AssimilatorsNum: The number of Assimilators (gas harvesting structures).
#   StargatesNum: The number of Stargates (air unit production buildings).
#   PylonsNum: The number of Pylons (supply and power-providing structures).
#   SupplyLeft: The amount of remaining supply capacity for building additional units.
#   SecondsOfGame: The elapsed game time in seconds.
OBSERVATION_SPACE_ARRAY = spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)

# Defines an empty observation indicating the initial or reset state of the environment,
# typically used at the start of a new episode.
EMPTY_OBSERVATION = np.zeros((224, 224, 3), dtype=np.uint8)

# Specifies the number of environments that are run in parallel during training.
# Running multiple environments concurrently can significantly speed up training
# by providing diverse experiences from multiple games.
NUMBER_OF_CONCURRENT_EXECUTIONS = 5

# Defines the number of timesteps for which the model is trained in each iteration.
# A timestep generally represents a single decision-making step of the agent.
TIMESTEPS = 10000

# The total number of iterations to run the training. Each iteration consists of
# training for the specified number of timesteps.
NUMBER_OF_ITERATIONS = 100000
