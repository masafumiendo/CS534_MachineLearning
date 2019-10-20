CS 534, Fall 2019
Group members: Masafumi Endo, Dilan Senaratne, Morgan Mayer

##########################
Part 0 run with: python part_0.py

The part_0.py file can be run and will output frequency distribution bar plots (png format) for each of the categorical features in the 'dist/' folder. It shows the percentage occurance in the train set for each catagory in each feature. The pandas DataFrame, df_num shows the mean, standard deviation, minimum, and maximum for each numerical feature. The data sets are scaled according to the Class, FeatureEngineering() which is in the file. It scales the train set, validation set, and test set based on the respective minimum and maximum values in the train set. The array of scaled train data are x_train, y_train with min and max being x_min and x_max respectively which are a pandas.Series object with feature index.

##########################
Part 1 run with: python part_1.py

The part_1.py file iterates through a list of learning rates and returns the calculated weights after the final epoch. Divergent models are associated with a True flag_div bool. Validation SSE is plotted for each case and output to figure_part1/ folder. Percent difference in predicted price for validation set is plotted as boxplots and output to the figure_part1/ folder.

##########################
Part 2 run with: python part_2.py

The part_2.py file iterates through a list of regularization parameters and returns the calculated weights after the final epoch. Divergent models are associated with a True flag_div bool. Validation SSE is saved in the dictionary dict_valid_sse and Train SSE is saved in the dictionary dict_train_sse. SSE vs. epoch plots are output to the figure_part2/ folder. Percent difference in predicted price for validation set is plotted as boxplots and output to the figure_part2/ folder. The pandas DataFrame df_weight is printed and includes the set of weights associated with the features for each regularization parameter. df_sse is also printed and contains the Train SSE and Validation SSE for all cases of regularization parameter and learning rate = 1e-5, determined as the best learning rate from part 1.

##########################
Part 3

##########################
