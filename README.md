# Jesse optuna

Only works with the new GUI version of Jesse.
It will create a CSV with the parameter set, training metrics and testing metrics. 
You can use that CSV for further evaluation. For example you might want to filter parameter sets where the testing metrics are too low / sort by best testing score etc.

The config.yml should be self-explainatory.

Check jesse-hyperactive.log for errors during optimization.

# Installation

```sh
# install from git
pip install git+https://github.com/cryptocoinserver/jesse-hyperactive.git

# cd in your Jesse project directory

# create the config file
jesse-hyperactive create-config

# edit the created yml file in your project directory 

# run
jesse-hyperactive run

```


## Disclaimer
This software is for educational purposes only. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS. Do not risk money which you are afraid to lose. There might be bugs in the code - this software DOES NOT come with ANY warranty.
