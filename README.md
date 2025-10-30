# Wavelet Transformer
This provides an easy-to-use interface for anyone to test out wavelet analysis on their time series data. The underlying algorithms are based off of my paper [Inflation expectations in time and frequency: A wavelet analysis](https://www.nathaniellawrence.com/research#h.z59n5ss724ja).

<i>Note: This version is still a prototype. Please, let me know what bugs, errors, etc. arise for you when using.</i>

## What is wavelet analysis?
[Wavelet](https://en.wikipedia.org/wiki/Wavelet) analysis allows us to explore the cyclical nature of time series, essentially extracting three dimensions of information from an otherwise one-dimensional dataset. Through the different wavelet transforms (continuous, CWT; wavelet coherence, WCT; and discrete, DWT), we can uncover and quantify patterns in the data that are imperceptible.

Normally, we would use a [Fourier transform](https://en.wikipedia.org/wiki/Fourier_transform) to identify the underlying cycles in a complex series, like in the image below. Effectively, the Fourier transform provides us a "fingerprint" of the underlying cyclical functions that produce the complex series in the aggregate. But, this only works for stationary data. In the case of nonstationary data, this fingerprints changes over time.

<i>Through a (continuous) wavelet transform, we can map how this fingerprint changes over time.</i>

### Example Fourier transform
![Example Fourier transform showing how simple cyclical functions can be identified in complex series](https://drive.google.com/uc?export=view&id=1sLj-vkNWcZBCWqG2aBdggVpgwgjAZwGW "Example Fourier transform")

### Example continous wavelet transform 
![Example continous wavelet transform showing how the composition of cyclical functions in a nonstationary time series changes over time](https://upload.wikimedia.org/wikipedia/commons/9/95/Continuous_wavelet_transform.gif "Example continous wavelet transform")

## Setting up the environment
To create a virtual environment using `conda`, use:
```
conda env create --file environment.yml
```
To create a virtual environment using `pip`, first, create virtual environment, named `venv`:
```
virtualenv venv
```
Activate `venv`:
```
venv\Scripts\activate
```
Install dependencies:
```
pip install -r requirements.txt
```
Once, you have created the environment, install the internal packages using:
```
pip install -e .
```

Finally, run the environment. For a `conda` virtual environment, use:
```
conda activate wavelet-transformer
```
For a `venv` generated via `pip`, use:
```
venv\Scripts\activate
```

## Running the dashboard locally
Simply run the follow command in your terminal:
```
streamlit run app.py
```
