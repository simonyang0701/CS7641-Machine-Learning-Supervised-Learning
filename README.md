# CS 7641 Assignment 1: Supervised Learning
## Author
Name: Tianyu Yang
GTid: 903645962
Date: 2020/9/20

## Introduction
This project is to generate five classification algorithms in supervised learning by using two interesting datasets downloaded from Kaggle
NBA games: https://www.kaggle.com/nathanlauga/nba-games
LOL games: https://www.kaggle.com/datasnaek/league-of-legends?select=games.csv

This project link on Github: https://github.com/simonyang0701/Supervised-Learning.git

From these two datasets, we will use some of the attributes of a game to generate a supervised learning model to analyze the result of a game. The five algorithms are including Decision Tree, Neural Networks, Boosting Tree, Support Vector Machines and K-nearest neighbors.

## Getting Started & Prerequisites
To test the code, you need to make sure that your python 3.6 is in recent update and the following packages have already been install:
pandas, numpy, scikit-learn, matplotlib, itertools, timeit


## Running the Classifiers
Recommendation Option: Work with the iPython notebook (.ipnyb) using Jupyter or a similar environment. Use "Run ALL" in Cell to run the code. Before running the code, make sure that you have already change the path into your current working directory
Another Option: Run the python script (.py) after changing the directory into where you saved the two datasets
Other Option (view only): Feel free to open up the (.html) file to see the output of program

The codes are divided into three parts:
1. Importing useful packages
2. Loading and cleaning datasets: load the two datasets and clean the datasets
3. Useful funciton: some useful functions we need in this program including machine learning algorithms part and plotting graphs part
4. Running machine learning models: run five supervised learning models and plot the graphs
5. Comparing with five supervised learning algorithms: compare them in three asepcts for two datasets, including F1 Score, training time and predicting time
