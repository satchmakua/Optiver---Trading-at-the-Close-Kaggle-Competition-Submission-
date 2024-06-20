Optiver - Trading at the Close Kaggle Competition

This repository contains the code for my solution to the Optiver - Trading at the Close Kaggle competition. The goal of the competition was to predict the closing price movements for Nasdaq listed stocks using data from the order book and the closing auction of the stock.

Getting Started

To run the code, you will need Anaconda installed on your machine. Anaconda is a package manager, an environment manager, and a Python distribution that contains a collection of open-source packages.

1. Install Anaconda from [here](https://www.anaconda.com/products/distribution).
2. Clone this repository to your local machine.
3. Open Anaconda Navigator and launch Jupyter Notebook.
4. Navigate to the directory where you cloned the repository and open the Jupyter Notebook file `optiver_trading_at_the_close.ipynb`.

Dependencies

The code is written in Python and uses the following libraries:

- pandas
- numpy
- xgboost
- sklearn

These dependencies can be installed using Anaconda by running the following commands in the Anaconda Prompt:

```
conda install pandas numpy xgboost scikit-learn
```

Running the Code

Once you have installed the dependencies, you can run the code in the Jupyter Notebook. Follow the instructions in the notebook to load the data, preprocess it, train the models, and make predictions.

Results

My model achieved a Mean Absolute Error (MAE) score of 5.4220 in the competition, ranking 2984 out of 3824 participants.

License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
