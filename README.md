# AL-HardnessRecommendation

![lc](/lc_1.png)
![overview](/overview.png)

## User Guide

![example](/output.png)

Run this line in your command prompt:

`python al_recommendation.py --pool_path /path/to/your/prediction_pool`

You will get the following message if the script runs without error:

![select](/select_strategy.png)

## Currently supported selection strategies (committee-based)

- [x] Query by Committee

committee models: Logistic Regression, Random Forest Regressor, Support Vector Regression, Gradient Boosting trees, XGB

![qbc](/QBC.png)

- [x] Query by bagging (entropy based)

draw with replacement

![qbb](/qbbagging.png)

- [x] Adaptive max disagreement

split subset of features

![amd](/qbfeatures.png)

## TBD

Margin-based strategies
