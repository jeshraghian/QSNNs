# Navigating Local Minima in Quantized Spiking Neural Networks
This paper contains the corresponding code from the paper *Navigating Local Minima in Quantized Spiking Neural Networks*. 

## To run
`cd` into one of three datasets, then `python run.py`.

## Hyperparameter tuning
* Hyperparameters are contained in `conf.py`. The default parameters in this repo are identical to those for the high precision case reported in the corresponding paper.
* To run 4-bit quantized networks, set `"num_bits" : 4"` in `conf.py`. For optimized parameters, follow the values reported in the paper (to be linked upon completion of double blind peer review process.)

## Plotting
* To enable plotting, set `plot_loss=True` in `conf.py`. 
* If you already have csv files containing loss values, then run `python plot_loss.py` directly
* As we store the loss at every iteration, you may wish to make the plot more visible. - This can be done by either:
    1. Downsampling by dropping data points. Set `drop_count=N` where `N` is an integer specifying the reduction factor, and uncomment `drop_count = 0`, `df = df[df['idx'] % drop_count == 0]` 
    2. Take a moving average by increasing `window` in `df['mean_rolling'] = df.iloc[:,3].rolling(window=20).mean()`.