{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "from sklearn.datasets import load_iris\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.cross_validation import cross_val_score\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Read the iris data\n",
    "iris = load_iris()\n",
    "\n",
    "# create X: features \n",
    "# create y: responses\n",
    "X = iris.data\n",
    "y = iris.target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 1.          0.93333333  1.          1.          0.86666667  0.93333333\n",
      "  0.93333333  1.          1.          1.        ]\n"
     ]
    }
   ],
   "source": [
    "# perform 10 fold cross validation to improve reliability of accuracy\n",
    "knn = KNeighborsClassifier(n_neighbors = 5)\n",
    "scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')\n",
    "print scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.966666666667\n"
     ]
    }
   ],
   "source": [
    "print scores.mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[0.95999999999999996,\n",
       " 0.95333333333333337,\n",
       " 0.96666666666666656,\n",
       " 0.96666666666666656,\n",
       " 0.96666666666666679,\n",
       " 0.96666666666666679,\n",
       " 0.96666666666666679,\n",
       " 0.96666666666666679,\n",
       " 0.97333333333333338,\n",
       " 0.96666666666666679,\n",
       " 0.96666666666666679,\n",
       " 0.97333333333333338,\n",
       " 0.98000000000000009,\n",
       " 0.97333333333333338,\n",
       " 0.97333333333333338,\n",
       " 0.97333333333333338,\n",
       " 0.97333333333333338,\n",
       " 0.98000000000000009,\n",
       " 0.97333333333333338,\n",
       " 0.98000000000000009,\n",
       " 0.96666666666666656,\n",
       " 0.96666666666666656,\n",
       " 0.97333333333333338,\n",
       " 0.95999999999999996,\n",
       " 0.96666666666666656,\n",
       " 0.95999999999999996,\n",
       " 0.96666666666666656,\n",
       " 0.95333333333333337,\n",
       " 0.95333333333333337,\n",
       " 0.95333333333333337]"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# refine cross_val_score by looking for the best value of knn\n",
    "# a for loop does this well\n",
    "k_range = range(1,31)\n",
    "k_scores = []\n",
    "\n",
    "# create a for loop and perform cross_val_score each time\n",
    "for k in k_range:\n",
    "    knn = KNeighborsClassifier(n_neighbors = k)\n",
    "    scores = cross_val_score(knn, X, y, cv = 10, scoring  = 'accuracy')\n",
    "    k_scores.append(scores.mean())\n",
    "k_scores\n",
    "# print doesn't go to new line so I dont include it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.text.Text at 0x1084230d0>"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x10831b590>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# print a graph to show relation btw k_scores for diff k\n",
    "# matplotlib makes this easy\n",
    "plt.plot(k_range, k_scores)\n",
    "plt.xlabel('knn with increasing k')\n",
    "plt.ylabel('score')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### There are 3 highest values at 13, 15 and 20. It is good practice to use the simplest mode."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from sklearn.grid_search import GridSearchCV"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]}\n"
     ]
    }
   ],
   "source": [
    "# define parameter scope for your GridSearch\n",
    "k_range = range(1,31)\n",
    "\n",
    "# create parameter grid\n",
    "param_grid = dict(n_neighbors=k_range)\n",
    "print param_grid"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "collapsed": false,
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GridSearchCV(cv=10, error_score='raise',\n",
       "       estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',\n",
       "           metric_params=None, n_jobs=1, n_neighbors=30, p=2,\n",
       "           weights='uniform'),\n",
       "       fit_params={}, iid=True, n_jobs=1,\n",
       "       param_grid={'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]},\n",
       "       pre_dispatch='2*n_jobs', refit=True, scoring='accuracy', verbose=0)"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# create a grid object\n",
    "grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')\n",
    "\n",
    "# fit the grid object with data\n",
    "grid.fit(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[mean: 0.96000, std: 0.05333, params: {'n_neighbors': 1},\n",
       " mean: 0.95333, std: 0.05207, params: {'n_neighbors': 2},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 3},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 4},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 5},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 6},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 7},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 8},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 9},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 10},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 11},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 12},\n",
       " mean: 0.98000, std: 0.03055, params: {'n_neighbors': 13},\n",
       " mean: 0.97333, std: 0.04422, params: {'n_neighbors': 14},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 15},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 16},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 17},\n",
       " mean: 0.98000, std: 0.03055, params: {'n_neighbors': 18},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 19},\n",
       " mean: 0.98000, std: 0.03055, params: {'n_neighbors': 20},\n",
       " mean: 0.96667, std: 0.03333, params: {'n_neighbors': 21},\n",
       " mean: 0.96667, std: 0.03333, params: {'n_neighbors': 22},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 23},\n",
       " mean: 0.96000, std: 0.04422, params: {'n_neighbors': 24},\n",
       " mean: 0.96667, std: 0.03333, params: {'n_neighbors': 25},\n",
       " mean: 0.96000, std: 0.04422, params: {'n_neighbors': 26},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 27},\n",
       " mean: 0.95333, std: 0.04269, params: {'n_neighbors': 28},\n",
       " mean: 0.95333, std: 0.04269, params: {'n_neighbors': 29},\n",
       " mean: 0.95333, std: 0.04269, params: {'n_neighbors': 30}]"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# view complete results in tuples\n",
    "grid.grid_scores_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'n_neighbors': 1}\n",
      "[ 1.          0.93333333  1.          0.93333333  0.86666667  1.\n",
      "  0.86666667  1.          1.          1.        ]\n",
      "0.96\n"
     ]
    }
   ],
   "source": [
    "# examining the tuples\n",
    "print grid.grid_scores_[0].parameters\n",
    "print grid.grid_scores_[0].cv_validation_scores\n",
    "print grid.grid_scores_[0]. mean_validation_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[0.95999999999999996,\n",
       " 0.95333333333333337,\n",
       " 0.96666666666666667,\n",
       " 0.96666666666666667,\n",
       " 0.96666666666666667,\n",
       " 0.96666666666666667,\n",
       " 0.96666666666666667,\n",
       " 0.96666666666666667,\n",
       " 0.97333333333333338,\n",
       " 0.96666666666666667,\n",
       " 0.96666666666666667,\n",
       " 0.97333333333333338,\n",
       " 0.97999999999999998,\n",
       " 0.97333333333333338,\n",
       " 0.97333333333333338,\n",
       " 0.97333333333333338,\n",
       " 0.97333333333333338,\n",
       " 0.97999999999999998,\n",
       " 0.97333333333333338,\n",
       " 0.97999999999999998,\n",
       " 0.96666666666666667,\n",
       " 0.96666666666666667,\n",
       " 0.97333333333333338,\n",
       " 0.95999999999999996,\n",
       " 0.96666666666666667,\n",
       " 0.95999999999999996,\n",
       " 0.96666666666666667,\n",
       " 0.95333333333333337,\n",
       " 0.95333333333333337,\n",
       " 0.95333333333333337]"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid_mean_score = [scores.mean_validation_score for scores in grid.grid_scores_]\n",
    "grid_mean_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.98\n",
      "{'n_neighbors': 13}\n",
      "KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',\n",
      "           metric_params=None, n_jobs=1, n_neighbors=13, p=2,\n",
      "           weights='uniform')\n"
     ]
    }
   ],
   "source": [
    "# examine the best model\n",
    "print grid.best_score_\n",
    "print grid.best_params_\n",
    "print grid.best_estimator_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Searching multiple parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# define parameter values for GridSearch\n",
    "k_range = range(1,31)\n",
    "weight_options = ['uniform', 'distance']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], 'weights': ['uniform', 'distance']}\n"
     ]
    }
   ],
   "source": [
    "# create parameter grid\n",
    "param_grid =  dict(n_neighbors = k_range, weights = weight_options)\n",
    "print param_grid"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GridSearchCV(cv=10, error_score='raise',\n",
       "       estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',\n",
       "           metric_params=None, n_jobs=1, n_neighbors=30, p=2,\n",
       "           weights='uniform'),\n",
       "       fit_params={}, iid=True, n_jobs=1,\n",
       "       param_grid={'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], 'weights': ['uniform', 'distance']},\n",
       "       pre_dispatch='2*n_jobs', refit=True, scoring='accuracy', verbose=0)"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid = GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy')\n",
    "grid.fit(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[mean: 0.96000, std: 0.05333, params: {'n_neighbors': 1, 'weights': 'uniform'},\n",
       " mean: 0.96000, std: 0.05333, params: {'n_neighbors': 1, 'weights': 'distance'},\n",
       " mean: 0.95333, std: 0.05207, params: {'n_neighbors': 2, 'weights': 'uniform'},\n",
       " mean: 0.96000, std: 0.05333, params: {'n_neighbors': 2, 'weights': 'distance'},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 3, 'weights': 'uniform'},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 3, 'weights': 'distance'},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 4, 'weights': 'uniform'},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 4, 'weights': 'distance'},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 5, 'weights': 'uniform'},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 5, 'weights': 'distance'},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 6, 'weights': 'uniform'},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 6, 'weights': 'distance'},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 7, 'weights': 'uniform'},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 7, 'weights': 'distance'},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 8, 'weights': 'uniform'},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 8, 'weights': 'distance'},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 9, 'weights': 'uniform'},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 9, 'weights': 'distance'},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 10, 'weights': 'uniform'},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 10, 'weights': 'distance'},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 11, 'weights': 'uniform'},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 11, 'weights': 'distance'},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 12, 'weights': 'uniform'},\n",
       " mean: 0.97333, std: 0.04422, params: {'n_neighbors': 12, 'weights': 'distance'},\n",
       " mean: 0.98000, std: 0.03055, params: {'n_neighbors': 13, 'weights': 'uniform'},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 13, 'weights': 'distance'},\n",
       " mean: 0.97333, std: 0.04422, params: {'n_neighbors': 14, 'weights': 'uniform'},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 14, 'weights': 'distance'},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 15, 'weights': 'uniform'},\n",
       " mean: 0.98000, std: 0.03055, params: {'n_neighbors': 15, 'weights': 'distance'},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 16, 'weights': 'uniform'},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 16, 'weights': 'distance'},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 17, 'weights': 'uniform'},\n",
       " mean: 0.98000, std: 0.03055, params: {'n_neighbors': 17, 'weights': 'distance'},\n",
       " mean: 0.98000, std: 0.03055, params: {'n_neighbors': 18, 'weights': 'uniform'},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 18, 'weights': 'distance'},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 19, 'weights': 'uniform'},\n",
       " mean: 0.98000, std: 0.03055, params: {'n_neighbors': 19, 'weights': 'distance'},\n",
       " mean: 0.98000, std: 0.03055, params: {'n_neighbors': 20, 'weights': 'uniform'},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 20, 'weights': 'distance'},\n",
       " mean: 0.96667, std: 0.03333, params: {'n_neighbors': 21, 'weights': 'uniform'},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 21, 'weights': 'distance'},\n",
       " mean: 0.96667, std: 0.03333, params: {'n_neighbors': 22, 'weights': 'uniform'},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 22, 'weights': 'distance'},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 23, 'weights': 'uniform'},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 23, 'weights': 'distance'},\n",
       " mean: 0.96000, std: 0.04422, params: {'n_neighbors': 24, 'weights': 'uniform'},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 24, 'weights': 'distance'},\n",
       " mean: 0.96667, std: 0.03333, params: {'n_neighbors': 25, 'weights': 'uniform'},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 25, 'weights': 'distance'},\n",
       " mean: 0.96000, std: 0.04422, params: {'n_neighbors': 26, 'weights': 'uniform'},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 26, 'weights': 'distance'},\n",
       " mean: 0.96667, std: 0.04472, params: {'n_neighbors': 27, 'weights': 'uniform'},\n",
       " mean: 0.98000, std: 0.03055, params: {'n_neighbors': 27, 'weights': 'distance'},\n",
       " mean: 0.95333, std: 0.04269, params: {'n_neighbors': 28, 'weights': 'uniform'},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 28, 'weights': 'distance'},\n",
       " mean: 0.95333, std: 0.04269, params: {'n_neighbors': 29, 'weights': 'uniform'},\n",
       " mean: 0.97333, std: 0.03266, params: {'n_neighbors': 29, 'weights': 'distance'},\n",
       " mean: 0.95333, std: 0.04269, params: {'n_neighbors': 30, 'weights': 'uniform'},\n",
       " mean: 0.96667, std: 0.03333, params: {'n_neighbors': 30, 'weights': 'distance'}]"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid.grid_scores_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.98\n",
      "{'n_neighbors': 13, 'weights': 'uniform'}\n"
     ]
    }
   ],
   "source": [
    "# examine best model\n",
    "print grid.best_score_\n",
    "print grid.best_params_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Using best parameters to make predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# train model on all data using best parameters\n",
    "knn = KNeighborsClassifier(n_neighbors = 13, weights = 'uniform')\n",
    "knn.fit(X, y)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
