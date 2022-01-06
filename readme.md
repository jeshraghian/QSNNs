# Quantized Spiking Neural Networks
This paper contains the corresponding code from the paper *Navigating Local Minima in Quantized Spiking Neural Networks*. 

## Requirements
A working `Python` (≥3.6) interpreter and the `pip` package manager. All required libraries and packages can be installed using  `pip install -r requirements.txt`. To avoid potential package conflicts, the use of a `conda` environment is recommended. The following commands can be used to create and activate a separate `conda` environment, clone this repository, and to install all dependencies:

```
conda create -n QSNNs python=3.8
conda activate QSNNs
git clone https://github.com/jeshraghian/QSNNs
cd QSNNs
pip install -r requirements.txt
```



## Code Execution
To execute code, `cd` into one of three dataset directories, and then run `python run.py`. 


## Hyperparameter Tuning
* In each directory, `conf.py` defines all configuration parameters and hyperparameters for each dataset. The default parameters in this repo are identical to those for the high precision case reported in the corresponding paper.
* To run 4-bit quantized networks, set `"num_bits" : 4"` in `conf.py`. For optimized parameters, follow the values reported in the paper (to be linked upon completion of the double blind peer review process.)

## Plotting
* To enable plotting, set `plot_loss=True` in `conf.py`. 
* If you already have csv files containing loss values, then `python plot_loss.py` can be run directly.
* As we store the loss at every iteration, you may wish to make the plot more visible. This can be done by either:
    1. Downsampling by dropping data points. Set `drop_count=N`, where `N` is an integer specifying the reduction factor, and uncomment `drop_count = 0`, `df = df[df['idx'] % drop_count == 0]` 
    2. Take a moving average by increasing `window` in `df['mean_rolling'] = df.iloc[:,3].rolling(window=20).mean()`.
