# LAI-Assignment-Group22

## About the project
This project contains the code used to get the results for our interim assignment for JBC090. The projects aim was to use 
stylometry on reddit posts to determine the writer's nationality. For this we have employed multiple prediction models, 
in this repository you'll find the code for all these models and the preprocessing techniques that were used to achieve 
this goal. 

## Dependencies
First of all install all packages from requirements.txt. You can do so for example using the pip command:

~~~
pip install -r requirements.txt
~~~
Secondly install jupyter, you can do so by following this tutorial: 
https://docs.jupyter.org/en/latest/install/notebook-classic.html

Lastly install java if you don't already have it installed. You can install the latest version from this website: 
https://www.oracle.com/java/technologies/downloads/

## Data

To ensure the input data is suitable for our functions we'll first describe the format of the data we got. Using other
formats for your data may result in having to rewrite significant parts of the code. The data we used were csv files 
with 3 columns: 
- author_id
  - Containing user ids used to differentiate between different authors.
- post
  - Containing the concatenation of the posts written by a certain author, one row containing at most 1500 words in this column.
- nationality
  - Containing the nationality of the author used as label.

There are 2 external datasets used for depollution, which can be downloaded here: 
- https://www.kaggle.com/datasets/juanmah/world-cities
- https://assets.publishing.service.gov.uk/media/5a81f5fce5274a2e87dc068d/CH_Nationality_List_20171130_v1.csv/preview.

## Files

### EDA
This repository includes three EDA files, each with a unique focus for analyzing the original dataset:

1. **`dataset_analysis.ipynb`**  
   - Performs data cleaning, preprocessing, and summary statistics.  
   - Visualizes distributions such as posts per nationality and language.  
   - Implements language detection to understand the language distribution.

2. **`EDA.py`**  
   - Focuses on pattern and context exploration.  
   - Includes functions to find specific words or phrases (e.g., "born in") and their surrounding context.  
   - Uses regex to identify and extract key text patterns efficiently.

3. **`visualizations_eds.py`**  
   - Specializes in word frequency analysis.  
   - Uses `CountVectorizer` to identify and plot the most frequently used words in the dataset.  
   - Provides insights into common textual patterns through visualizations.

### Data preparation
These are the files to prepare your data before running the models, here we'll go through all of them in the order which
they should be run. Do note that the only thing absolutely necessary to run the models, is to use the tokenize function 
in create_tokens.py. However, these files do contain different ways of preparing the data, which might be nice to use.
1. Create_subset_of_data.py
   - Can be used to create a subset of your data, this may be nice to use for experimentation purposes to reduce runtime.
3. Depollution.py
   - Can be used to depollute the data a bit, by removing things like explicit mentions of nationalities, countries or cities.
5. correct grammar.py and/or experiment_3_data_preparation.ipynb and/or experiment_4_data _preparation.ipynb
   - Any of these 3 can be run after running Depollution.py.
   - correct grammar.py corrects the grammar of the text after certain words got removed by other cleaning functions.
   - experiment_3_data_preparation.ipynb replaces foreign words with a special token.
   - experiment_4_data _preparation.ipynb removes all non english posts.
7. create_tokens.py
   - Tokenizes the text and can be used to standardise the text further.

### Models
The models can all be run after running the files mentioned in the previous section (data preparation). In these files 
make sure to change the file path of the train and test data to as needed.
There are 6 files that each contain a prediction model and are named after the model it contains, these files are:
- baseline_counter_heuristic.py
- linear_regression.py
- linear_regression_with_smote.py
- Logistic_Regression.py
- Logistic_Regression_SMOTE.py
- svm_implementation.ipynb

You'll see that linear and logistic regression each occur twice, once with smote and once without. Smote is an 
oversampling technique used to create synthetic data samples of minority classes. Thus, in the one without smote the 
dataset is not being artificially balanced and in the one with smote it is.
