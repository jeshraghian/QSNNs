# Quantized Spiking Neural Networks
This repository contains the corresponding code from the paper [Jason K. Eshraghian, Corey Lammie, Mostafa Rahimi Azghadi, and Wei D. Lu "Navigating Local Minima in Quantized Spiking Neural Networks". https://arxiv.org/abs/2202.07221, February 2022.](https://arxiv.org/abs/2202.07221)


![anim_2](https://user-images.githubusercontent.com/40262130/154583824-fa940d58-3249-40aa-a85b-0c0fbcaf68c4.gif)

<p style="text-align: center; align="center"><i>Illustrations of the key concepts of the paper: Periodic scheduling can enable SNNs to overcome flat surfaces and local minima. When the LR is boosted during training using a cyclic scheduler, it is given another chance to reduce the loss with different initial conditions. While the loss appears to converge, subsequent LR boosting enables it to traverse more optimal solutions</i>.</p>

If you find this code useful in your work, please cite the following source:

```
@article{eshraghian2022navigating,
  title={{Navigating Local Minima in Quantized Spiking Neural Networks}},
  author={Eshraghian, Jason K and Lammie, Corey and Rahimi Azghadi, Mostafa and Lu, Wei D},
  year={2022},
  eprint={2202.07221},
  archivePrefix={arXiv},
}
```

## Jupyter Notebook
We provide a Jupyter notebook [here](https://github.com/jeshraghian/QSNNs/blob/main/quickstart.ipynb), which includes documentation and information about our developed scripts and methodologies. This can be run in a Google Collaboratory environment without any prerequisites [here](https://colab.research.google.com/github/jeshraghian/QSNNs/blob/main/quickstart.ipynb).

## Code Execution of Standalone Scripts 
For more advanced users, i.e., those proficient with Python, we provide executable code in the form of Python scripts. Simulations can be run by configuring and executing `run.py` in each respective dataset directory.

## Requirements
### Jupyter Notebook
To run the Jupyter notebook, Google Colab can be used. Otherwise, a working `Python` (≥3.6) interpreter and the `pip` package manager are required.

### Standalone Scripts
To run all standalone scripts, a working `Python` (≥3.6) interpreter and the `pip` package manager. All required libraries and packages can be installed using  `pip install -r requirements.txt`. To avoid potential package conflicts, the use of a `conda` environment is recommended. The following commands can be used to create and activate a separate `conda` environment, clone this repository, and to install all dependencies:

```
conda create -n QSNNs python=3.8
conda activate QSNNs
git clone https://github.com/jeshraghian/QSNNs.git
cd QSNNs
pip install -r requirements.txt
```

## Hyperparameter Tuning
* In each directory, within `run.py` files, the `config` dictionary defines all configuration parameters and parameters for each dataset. 
* The default parameters in this repo are identical to those for the Q4 cosine anneling learning rate schedule configurations reported in the corresponding paper.

## Interpreting and Plotting Results
* Results can be gathered and plotted using `extract_test_set_accuracy.py` and `plot_results.py`, respectively. 
* `plot_results.py` can be reconfigured to plot different quantities.
* By default, `plot_results.py` plots the loss curve evolution during training for all three datasets.
