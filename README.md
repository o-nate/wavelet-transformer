# Wavelet Dashboard
This provides an easy-to-use interface for anyone to test out wavelet analysis on their time series data. The underlying algorithms are based off of my paper [Inflation expectations in time and frequency: A wavelet analysis](https://www.nathaniellawrence.com/research#h.z59n5ss724ja).

<i>Note: This version is still a prototype and quite brittle. Please, let me know what bugs, errors, etc. arise for you when using.</i>

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
