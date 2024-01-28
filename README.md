![Static Badge](https://img.shields.io/badge/test-jupyter-blue?logo=jupyter&label=code&labelColor=grey&color=orange)
![Static Badge](https://img.shields.io/badge/test-python-blue?logo=python&label=code&labelColor=grey&color=blue)

![Static Badge](https://img.shields.io/badge/test-github-blue?logo=github&label=Tools&labelColor=grey&color=black)


<h1>Improving the general optimization strategy with a data-driven framework</h1>


<h2>Description</h2>

The project aims to design a machine learning-based optimization strategy that competes with conventional optimizers. By assessing the performance of various opimizers on benchmark functions a strategy using the f3dasm framework was developed. For the data-driven framework machinelearning and deeplearning approaches were analyzed. 

<h3>Table of contents</h3>

- How to install and run the project
- Project roadmap
- How to use the project
  - Machine learning
  - Deep learning
- Credits
- Contributing
- License



How to install and run the project
--
To install the project use:

```
pip install -r "requirements.txt"
```

This will install all the required packages. The main code is available in the file. Import your data and then run all the code.
This will give you the results.


<h2>Project roadmap</h2>

* (<b>Done</b>) Must read the provided literature on optimization and learning to optimize.
* (<b>Done</b>) Must implement a data-driven optimization strategy model to choose an optimization algorithm for a wide variety of tasks.
* (<b>Done</b>) Must use the datasets provided to train their data-driven optimization strategy model.
* (<b>Done</b>) Must use the f3dasm framework and Python helper-function coding interface file provided.
* (<b>Done</b>) Should do a hyper-parameter study on the chosen data-driven optimization strategy model.
* (<b>Done</b>) Should investigate the dataset to reveal the pros and cons of the optimizers on the benchmark functions.
* (<b>Done</b>) Could do a sensitivity analysis of their data-driven optimization strategy on sub-sets of the training data.
* Won’t implement their own benchmarks or simulation data.
* Won’t optimize the benchmark functions with a self-chosen optimizer.
* Won’t do experimental research on a materials science topic to produce data.

<h2>How to use the project</h2>

For both ways of approach the file: l2o.py was used.

<h3>Machine learning</h3>
For machine learning there are the following directories and files:

* ML_main.ipynb
* ML_tuning.ipynb
* tools
  * DataLoading.py
  * CustomStrategies.py
  * ModelEvaluation.py
  * l2o_modified.py
* studies
* models

Tuning of the final model can be found in *ML_tuning.ipynb*. Experimenting and development of the models was done in *ML_main.ipynb*. We modified *l2o.py* in *l2o_modified.py* to correct the best and worst performance profile calculation. The rest of the *.py* files in tools are tools used to load data, define custom strategies, and evaluate the models. In *studies*, you can find results of optuna hyperparameter searches from *ML_tuning.ipynb*. In *models*, you can find the saved models, saved using *joblib.dump()*.

<h3>Deep learning</h3>
For deep learning there are the following directories and files:

* DL_main.ipynb
* DL_tuning.ipynb
* tools
  * DatasetExtraction.py
  * l2o.py
  * UsefulFunctions.py
* ready_files

To go through the process of DL project it is enough to run the following notebooks in order:

1. DL_tuning.ipynb
2. DL_main.ipynb

The above notebooks use helper functions from scripts stored in *tools* directory. *DatasetExtraction.py* stores functions that extract data needed for model training and making predictions from provided files with datasets. In *UsefulFunctions.py* other helper functions and classes are defined, which are then used in the notebooks. In *ready_files* folder some files with quick access data are stored. They include extracted datasets along with their features and labels, which were used in training. Furthermore, the final model, consisting of a neural network and a scaler, is stored here.

<h3>Final product</h3>

The final product is delivered in *main.py* in the directory *Scripts*. It uses the best model that has been found and makes predictions on provided input data. To generate predictions input data should be put in the *Input* folder. Data should be stored in a directory, which consists another directory called *post*. In the *post* folder problems, for which predictions are to be made, should be put in *.nc* files. Then, in *main.py* *INPUT_DIR* variable needs to refer to the directory with the *post*. After that, *main.py* should be run and after it has finished running, the predictions are saved in the file *Results.csv* in *Results* directory. Name of the file with predictions may be changed by changing *OUTPUT_DIR* variable.

<h2>Credits</h2>
Leon Dybioch, Daniel Tap, Anton Belitser, Ralph Tamming, Maurits Rietvelt, Martin van der Schelling

<h2>Contributing</h2>
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

<h2>License</h2>

[GNU](https://choosealicense.com/licenses/gpl-3.0/)
 

