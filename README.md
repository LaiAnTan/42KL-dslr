# 42KL-dslr

A project to learn about data cleaning, data visualisation, logistic regression and multiclass classification by recreating Poulard's Sorting Hat from Harry Potter, i.e. using the One VS Rest strategy with Logistic Regression to classify Hogwarts students using academic data.

## Dependencies

Dependencies listed in [requirements.txt](/requirements.txt)

## Installation

1. `git clone https://github.com/LaiAnTan/42KL-dslr`
2. `pip install -r requirements.txt` or in other environments such as conda

## Usage

Datasets can be found in the `datasets` directory.

Data Analysis:

`python3 srcs/describe.py <dataset_train / dataset_test>` prints a statistical report onto the terminal.

Data Visualisation:

`python3 srcs/histogram.py` saves a histogram to `graphs` directory.

`python3 srcs/scatter_plot.py` saves a scatter plot to `graphs` directory.

`python3 srcs/pair_plot.py` saves a pair plot to `graphs` directory.

ML Model:

`python3 srcs/logreg_train.py <dataset_train>` trains the model and saves the model details to `models` directory.

`python3 srcs/logreg_plot.py <dataset_test>` loads the model and generates a file with predictions, `dataset_prediction.csv` to `datasets` directory.

## Notes

Notes written in [notes.md](/notes.md).
