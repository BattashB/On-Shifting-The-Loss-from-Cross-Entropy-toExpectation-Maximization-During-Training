# CINIC10
This dataset is a dataset which is smaller then imagenet but larger the cifar100.
It holds 270,000 images, 90k for each split, we combined the train and validation subsets in order to make a larger training set.
The dataset can be downloaded from: [Official repo](https://github.com/BayesWatch/cinic-10), you can also find the script that combines the the splits in that repo.

You can run the entire experiments for each model using:
```
python3 run_model.py <model_name> <data_path>
```
