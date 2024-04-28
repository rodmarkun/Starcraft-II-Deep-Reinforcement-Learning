# Starcraft II - Deep Reinforcement Learning

Starcraft II is a popular real-time strategy (RTS) game developed by Blizzard Entertainment, released in 2010. It is the sequel to the 1998 game Starcraft, and like its predecessor, it has had a significant impact on both the gaming and AI research communities. The game is set in a distant part of the galaxy during the 26th century and features three different species: Terrans, Protoss, and Zerg, each with unique units and structures, offering varied strategies and gameplay styles.

Starcraft II is a complex real-time strategy game that has become a significant benchmark in the field of deep reinforcement learning (DRL) due to its intricate decision-making requirements and dynamic gameplay. The game's environment, which demands real-time actions and decisions across diverse strategies and continuous adjustments to opponents' tactics, challenges DRL models to handle real-world-like scenarios with high levels of adaptability and strategic depth. 

This project is a deep reinforcement learning training software designed for Starcraft II. It utilizes StableBaselines3 and OpenAI Gym to create a robust training environment that allows users to train AI models to play Starcraft II at a competitive level. This README provides an overview of the software, including setup instructions, usage details, and contributions guidelines.

## Features
- Training AI agents to play Starcraft II using StableBaselines3 for advanced model training.
- Custom Starcraft II environment supported by OpenAI Gym for realistic and configurable gameplay scenarios.
- Saving and loading of trained models supporting curriculum learning.

## Installation and Usage

> **NOTE**: It is VERY recommended to use the trainer in a Linux system, as it can run the game in *headless* mode (game is not rendered) and thus train way faster. The Linux binary also already comes with pre-installed maps, so you will not need to install them yourself.

### Installation

As prerequisites to the installation, you will need to have Python 3.10 and Git installed on your machine. To get started with the Starcraft II DRL Trainer, follow these installation steps:

1. Install the [Starcraft II Linux Binary](https://github.com/Blizzard/s2client-proto#downloads) and extract it in your root. Check that it has maps installed under the `maps/` folder. If the `maps/` folder does not exist, create it and download maps from [here](https://github.com/Blizzard/s2client-proto?tab=readme-ov-file#map-packs).
2. Clone this repository by using:

```
git clone https://github.com/rodmarkun/Starcraft-II-Deep-Reinforcement-Learning && cd Starcraft-II-Deep-Reinforcement-Learning
```

3. Install necessary dependencies:

```
pip install -r ./requirements.txt
```

4. Finally, you will need to **comment two lines** in the burnysc2 API. These lines are located in `sc2/sc2process.py`. They are both SIGINT senders and their current line numbers in version 6.5.0 are lines 104 and 119.

```python
# LINE 104:
signal.signal(signal.SIGINT, signal_handler)
# LINE 119:
signal.signal(signal.SIGINT, signal.SIG_DFL)
```

5. Now, you can customize your DRL agent configuration in `constants.py` or also in `main.py`. Just execute `main.py` to begin training your agents.

## Contributing

We welcome contributions to the Starcraft II DRL Trainer. If you have suggestions or improvements, please follow these steps:

- Fork the repository.
- Create a new branch (git checkout -b feature-branch).
- Make your changes.
- Commit your changes (git commit -am 'Add some feature').
- Push to the branch (git push origin feature-branch).
- Create a new Pull Request.

## Contact

Pablo Rodríguez Martín - rodmarprogrammer@gmail.com