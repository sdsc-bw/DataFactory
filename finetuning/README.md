# Fine-tuning
## What is Fine-tuning

When developing a model, we use different models with different hyperparameters, like the maximal depth of a decision tree or the numbers of trees in a random forest. To get a good model, we use fine-tuning to find the best model and according to that the best hyperparameters of a model for a given problem.

## Proposed methods
There is a variety of fine-tuning frameworks and we want to present the most common methods in this Demo. We propose [Hyperopt](http://hyperopt.github.io/hyperopt/), [Auto-sklearn](https://automl.github.io/auto-sklearn/master/) and [Sklearn with model selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection). Auto-sklearn automatically chooses a suitable model, but it only runs on linux systems or colab. Hyperopt and sklearn tune the models that are chosen.
