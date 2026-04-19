# Chaos Branch Experiment: State Space Modeling of the Lorenz System

## Overview
This branch contains an experiment dedicated to modeling and predicting chaotic dynamics using State Space Models (SSMs). Specifically, we generate data from the chaotic **Lorenz-63 system** and train an SSM to learn its continuous-time dependencies and predict its future trajectory.

## Project Contents
- **`chaos.ipynb`**: The primary Jupyter notebook for the experiment. It contains the data generation logic (Runge-Kutta integration), model architecture definition, and the PyTorch training loop.
- **`images/`**: Generated visualizations from the model's training and evaluation:
  - `actual_lorenz.png`: The ground truth trajectory of the generated Lorenz system.
  - `output_lorenz.png` / `3D_output_lorenz.png`: The predicted trajectories output by the trained State Space Model.
  - `A_matrix.png`: A visual representation of the transition matrix (typically a HiPPO matrix) which enables the SSM to memorize long-range dependencies.
  - `gradient_plot.png`: Visualization of the loss/gradients over training to monitor stability.

## Methodology
1. **Data Generation**: We simulate the Lorenz-63 differential equations for 100,000 steps using the 4th-order Runge-Kutta (`rk4_step`) method.
2. **Preprocessing**: The trajectory is normalized and restructured into sliding windows (sequence length `L=50`) to create training and validation sets (with an 80/20 split).
3. **Modeling**: A State Space Model is trained on the sliding windows. The SSM excels at capturing continuous temporal dynamics, making it well-suited for chaotic differential equations.
4. **Evaluation**: The model is evaluated by predicting future states, resulting in a reconstructed 3D chaotic attractor.

## Prerequisites
To run the notebook locally, ensure you have Python installed along with the following packages:
- `torch`
- `numpy`
- `matplotlib`
- `plotly`
- `seaborn`
- `scipy`
- `jupyter` (or `jupyterlab`)

You can install all required dependencies with:
```bash
pip install torch numpy matplotlib plotly seaborn scipy jupyter
```

## Running the Experiment
1. Clone the repository and switch to the `chaos` branch (or clone your fork):
   ```bash
   git clone -b chaos https://github.com/DeeptamBhar/Team1_AIML.git
   cd Team1_AIML
   ```
2. Open the notebook in Jupyter:
   ```bash
   jupyter notebook chaos.ipynb
   ```
3. Run all cells from top to bottom to regenerate the data, train the State Space Model, and view the output visualizations.
