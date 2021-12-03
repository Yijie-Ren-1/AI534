There are 3 python files -- IA4_decision_tree.py, IA4_random_forest.py and IA4_adaboost.py, which refers to Part 1, Part 2 and Bonus Part, respectively. The result of IA4_decision_tree.py contains the console output, indicating the best feature & best gain for different depth of the tree, as well as the corresponding training/validation accuracies. The results of the other 2 python files are figures, and the console output is the number of depth and feature selections, if applicable. I've attached all the required figures to the report, and all figures will also be stored in a folder named "plots" after running different python files.

1. Please activate python3 virtual environment in the same directory where the python files and the data are. Please make sure the data files and python files are under same folder.
2. Please install these python3 packages under the virtual environment: pandas, numpy, matplotlib and seaborn. 
3. To run the program, please use the following command: python3 IA4_decision_tree.py, python3 IA4_random_forest.py, python3 IA4_adaboost.py, for different parts. 

Please note that when dmax=5, IA4_adaboost.py will train for a very long time, approx. longer than 40 mins. If you do not have enough time, you can change the "dmax_list" variable in the main function to have less dmax values.

I've tested on the babylon server, the python files should work~^_^