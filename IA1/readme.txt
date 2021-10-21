There is only one python file, which can generate the lineplots for different learning rates. Please change the "learning_r" variable in main function to see different results.  The lineplots will be save in the path defined by variable "MSE_plot_save_path" in main function. The lineplots are generated separately. Each execution of python file will also print the MSE value for validation set, and save the prediction value of validation set in directory './val_prediction.csv'

If you want to check the results for Kaggle, please set variable "is_kaggle_data" to "True", and also uncomment sections in main function and function "feature_normalization". Also uncomment the line "y_predicted[y_predicted < 0] = Y_train.min()" in the main function.

1. Please activate python3 virtual environment in the same directory where the python file and the data are. Please make sure the data files and python file are under same folder.
2. Please install these python3 packages under the virtual environment: pandas, numpy, matplotlib and seaborn. 
3. To run the program, please use the following command: python3 IA1_modified.py

I've tested on the babylon server, so the python file should work~^_^

