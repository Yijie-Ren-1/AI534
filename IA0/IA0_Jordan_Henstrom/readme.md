

READ THIS: I had issues working with a traditional virtual env, so I am using an anaconda venv. As such, the below
is how to do that. If you are able to get a normal venv set up with numpy, pandas, and matplotlib. Then my code should work for that. So if that is the case you can skip to step 3.

Step 0: Set up anaconda (if not set up for you, else skip to step 1):

> wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
> bash Anaconda3-2020.07-Linux-x86_64.sh
> echo 'export PATH="~/anaconda3/bin:$PATH"' >> ~/.bashrc
> cd ~
> source .bashrc
> conda update conda

Step 1: Create a python virtual env

> conda create --name IA0
> conda install -n IA0 pip
> conda activate IA0

Step 2: verify the following packages are installed:
    1. numpy
    2. pandas
    3. matplotlib

> pip install numpy
> pip install pandas
> pip install matplotlib

Step 3: running the program:

> cd <PATH_TO_WHERE_YOU_DOWNLOADED_ZIP>/IA0_Jordan_Henstrom
> python IA0.py data.csv

Figures can be found in the pdf, or as generated pngs in the folder