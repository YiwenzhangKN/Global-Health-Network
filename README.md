# FS-MADS-IDA Global Health Network

## How to use:

The repository contains the viewer, data and models. All final modelling outputs are stored within the data folder, including all data needed to run the script. To run the code, the following steps are required:

1. Ensure data from the sources is within data/raw folder
2. python dataset_builder.py - Build the initial dataset
3. (OPTIONAL) python initial_eda.py - To gain data insights and understand the dataset
4. python clean_dataset.py - To clean the dataset
5. (OPTIONAL) eda.ipynb - Gain insight into cleaned dataset
6. python model.py to run the model

## Launching Web Viewer:

1. Navigate to the "viewer" folder
2. Run "npm install" to install node modules, ensure that you have node.js installed
3. Create an .env file based on the example .env.example file, where the port can be set
4. Use node server.js to launch the viewer
5. Open the link from the console in your browser on your local machine 

## Issues or further:
The following versions where used to run the code and viewer:
- Python 3.12.10
- Node.js v20.19.0
- NPM 10.8.2