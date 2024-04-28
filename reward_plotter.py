import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('models/1714143864/episode_rewards.csv')

# Plot the data
plt.figure(figsize=(15, 6))  # Set the figure size
plt.plot(data['Total Episode Reward'], marker='o', linestyle='-')  # Line plot with markers
plt.title('Total Episode Rewards Over Episodes')  # Title of the plot
plt.xlabel('Episode')  # X-axis label
plt.ylabel('Total Episode Reward')  # Y-axis label
plt.grid(True)  # Enable grid
plt.show()  # Display the plot
