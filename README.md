# SIMS
Additional materials for paper "Modeling User Interests With Online Social Network Influence by Memory Augmented Sequence Learning" accepted by TNSE 2021.
## :page_facing_up: Description
SIMS is a novel social-based sequence learning model for predicting the types of items/PoIs that a user will likely buy/visit next.
Specifically, SIMS leverages the sequence-to-sequence learning method to learn a representation for each user sequence.
Moreover, an autoencoder-based model was proposed to learn social influence, which is integrated into SIMS for predicting user interests.
In addition, SIMS employs DNC to further improve prediction accuracy.
## :wrench: Dependencies
- Python == 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch == 1.8.1](https://pytorch.org/)
- NVIDIA GPU (RTX 3090) + [CUDA 11.1](https://developer.nvidia.com/cuda-downloads)
### Installation
1. Clone repo
    ```bash
    git clone https://github.com/BIT-MCS/SIMS.git
    cd SIMS
    ```
2. Install dependent packages
    ```
    pip install -r requirements.txt
    ```
## :computer: Training

We provide complete training codes for SIMS.<br>
You could adapt it to your own needs.

1. If you don't have NVIDIA RTX 3090, you should comment these two lines in file
[SIMS/main.py](https://github.com/BIT-MCS/SIMS/main.py).
	```
	[17]  torch.backends.cuda.matmul.allow_tf32 = False
	[18]  torch.backends.cudnn.allow_tf32 = False
	```
2. You can modify the config files 
[SIMS/conf.py](https://github.com/BIT-MCS/SIMS/conf.py) for model training.<br>
For example, you can control the size of RNN in the model by modifying this line
	```
	[8]  'rnn_size': 128,
	```
3. Training
	```
	cd SIMS
	python main.py
	```
	The log files will be stored in [SIMS/log].
## :checkered_flag: Testing
1. Testing
	```
	cd SIMS
	python test.py
	```
## :e-mail: Contact

If you have any question, please email `2656886245@qq.com`.
