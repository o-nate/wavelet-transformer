# Wavelet Dashboard

## Setting up the environment
To create a virtual environment using `conda`, use:
```
conda env create --file environment.yml
```
To create a virtual environment using `pip`, use:
```
pip install -r requirements.txt
```

Once, you have created the environment, install the internal packages using:
```
pip install -e .
```

Finally, run the environment. For a `conda` virtual environment, use:
```
conda activate wavelet-dashboard
```
For a `venv` generated via `pip`, use:
```
venv\Scripts\activate.bat
```

## Running the dashboard locally
Simply run the follow command in your terminal:
```
streamlit run app.py
```