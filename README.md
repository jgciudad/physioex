<div style = "text-align: center;">
<img src="https://raw.githubusercontent.com/guidogagl/physioex/refs/heads/main/docs/assets/images/logo.svg" width = "250px", alt="PhysioEx Logo">

<h1> PhysioEx </h1>
</div>

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![PyPI Version](https://badge.fury.io/py/physioex.svg)

**PhysioEx ( Physiological Signal Explainer )** is a versatile python library tailored for building, training, and explaining deep learning models for physiological signal analysis. 

The main purpose of the library is to propose a standard and fast methodology to train and evalutate state-of-the-art deep learning architectures for physiological signal analysis, to shift the attention from the architecture building task to the explainability task. 

With PhysioEx you can simulate a state-of-the-art experiment just running the `train`, `test_model`  and `finetune` commands; evaluating and saving the trained model; and start focusing on the explainability task! 

## Supported deep learning architectures

- [Chambon2018](https://ieeexplore.ieee.org/document/8307462) model for sleep stage classification ( raw time series as input).
- [TinySleepNet](https://github.com/akaraspt/tinysleepnet) model for sleep stage classification (raw time series as input).
- [SeqSleepNet](https://arxiv.org/pdf/1809.10932.pdf) model for sleep stage classification (time-frequency images as input).

## Supported datasets

- [SHHS (Sleep Heart Health Study)](https://sleepdata.org/datasets/shhs), is a multi-center cohort study designed to investigate the cardiovascular and other consequences of sleep-disordered breathing. At visit 1, it included 5,793 participants aged 40 years or older. PSG recordings were typically conducted in the subjects' homes by trained and certified technicians. The recording montage included C3/A2 and C4/A1 EEGs sampled at 125 Hz, right and left EOGs sampled at 50 Hz, and a bipolar submental EMG sampled at 125 Hz.
- [MASS (Montreal Archive of Sleep Studies)](http://ceams-carsm.ca/mass/), is an open-access collaborative database containing laboratory-based PSG recordings. It includes 200 complete nights recorded from 97 men and 103 women, aged 18 to 76 years. All recordings have a sampling frequency of 256 Hz and feature an EEG montage of 4–20 channels, along with standard EOG and EMG
- [MESA (Multi-Ethnic Study of Atherosclerosis)](https://sleepdata.org/datasets/mesa), is a multi-center prospective study of 2.237 ethnically diverse men and women aged 45-84 from six communities in the United States. PSGs recordings were obtained using in-home settings including central C4-M1 EEG, bilateral EOG and chin EMG sampled at 256Hz. PSGs were scored by one of 3 MESA certified, registered polysomnologists.
- [MrOS (The Osteoporotic Fractures in Men Study)](https://sleepdata.org/datasets/mros), is a multicenter study comprising 2,911 PSG recordings from men aged 65 years or older, enrolled at six clinical centers. PSG recordings were conducted in home settings and included C3/A2 and C4/A1 EEGs, chin EMG, and left-right EOG, all sampled at 256 Hz.
- [HMC (Haaglanden Medisch Centrum)](https://physionet.org/content/hmc-sleep-staging/1.1/), is a collection of 151 whole-night PSG recordings from 85 men and 66 women, gathered at the Haaglanden Medisch Centrum sleep center. The PSG data includes 4 EEG channels (F4/M1, C4/M1, O2/M1, and C3/M2), two EOG channels (E1/M2 and E2/M2), and one bipolar chin EMG, with all signals sampled at 256 Hz.
- [DCSM (Danish Center for Sleep Medicine)](https://erda.ku.dk/public/archives/db553715ecbe1f3ac66c1dc569826eef/published-archive.html), is a collection of 255 randomly selected and fully anonymized overnight lab-based PSG recordings from patients seeking diagnosis for non-specific sleep-related disorders at the DCSM. The PSG setup included EEG, EOG, and EMG channels, all sampled at 256 Hz.

For the public available datasets ( DCSM, HMC ) PhysioEx takes care of automatically download the data thanks to the `preprocess` command. The other datasets needs to be acquired first ( mostly on [NSSR](https://sleepdata.org) ) and then fetched by PhysioEx via the `preprocess` command.

## Installation guidelines

### Create a Virtual Environment (Optional but Recommended)

```bash
    conda create -n physioex python==3.10
    conda activate physioex
    conda install pip
    pip install --upgrade pip  # On Windows, use `venv\Scripts\activate`
```
### Install via pip

1. **Install PhysioEx from PyPI:**
```bash
pip install physioex
```

### Install from source
1. **Clone the Repository:**
```bash
   git clone https://github.com/guidogagl/physioex.git
   cd physioex
```

2. **Install Dependencies and Package in Development Mode**
```bash
    pip install -e .
```


## Cite Us!
```bib
@article{10.1088/1361-6579/adaf73,
	author={Gagliardi, Guido and Alfeo, Luca and Cimino, Mario G C A and Valenza, Gaetano and De Vos, Maarten},
	title={PhysioEx, a new Python library for explainable sleep staging through deep learning},
	journal={Physiological Measurement},
	url={http://iopscience.iop.org/article/10.1088/1361-6579/adaf73},
	year={2025},
}
```
