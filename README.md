<p align="center">
  <a href="" rel="noopener">
  <img width=200px height=200px src="assets/py-grex-logo.png" alt="Project logo"></a>
</p>

<h3 align="center">PY-GREX</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]() 
[![GitHub Issues](https://img.shields.io/github/issues/toledomateus/py-grex.svg)](https://github.com/toledomateus/py-grex/issues) 
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/toledomateus/py-grex.svg)](https://github.com/toledomateus/py-grex/pulls) 
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE) 

</div>

---

<p align="center"> A software toolkit for explainable recommender systems, including several state-of-the-art explainability methods and evaluation metrics.
    <br> 
</p>

## 📝 Table of Contents
- [🧐 About ](#-about-)
- [🏁 Getting Started ](#-getting-started-)
  - [Prerequisites ](#-prerequisites-)
  - [Installing  ](#-installing-)
- [⛏️ Built Using ](#️-built-using-)
- [✍️ Authors ](#️-authors-)



## 🧐 About <a name = "-about-"></a>

Recommender systems heavily shape our digital experiences and decision-making processes Consequently, various parties involved require insight into how these systems generate predictions.

This demand for explainability is multifaceted, ranging from individual user rationales to the more complex challenge of **recommender group explainability**, which seeks to clarify recommendations made for groups of users.

Recognizing that explanations can enhance trust, efficiency, and even persuasive power, researchers have actively pursued this area. Yet, despite this surge in interest, the field currently lacks a standard, accessible toolkit for implementing and evaluating explainable recommendation techniques, especially for the nuanced domain of group settings. Researchers often find themselves bogged down re-implementing established methods
 
Addressing this critical need, we introduce PY-GREX, a toolkit equipped with multiple state-of-the-art explainability algorithms to facilitate progress in the field.


## 🏁 Getting Started <a name = "-getting-started-"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### Prerequisites <a name = "-prerequisites-"></a>

What things you need to install the software and how to install them.

The following toolkits are necessary:
- conda
- git
- 

```bash
# Example: Check if conda is installed
conda --version

# Example: Check if git is installed
git --version 
```

### Installing <a name = "-installing-"> </a>


A step by step series of examples that tell you how to get a development environment running.

Clone the repo:
```bash
git clone https://github.com/toledomateus/pygrex.git
```

Navigate into the cloned directory

```
cd pygrex 
```

Create environment on conda (PY-GREX was developed with python 3.11):

```
conda create -n pygrex python=3.11
``` 

Activate the new environment:

```
conda activate pygrex
```

Install PyTorch as explained in https://github.com/pytorch/pytorch#from-source. The version used during development was without CUDA support.

Choose the version appropriate for your system (CPU or specific CUDA version)

When PyTorch is installed, navigate to the folder where you cloned the library and install PY-GREX and its dependencies in editable mode:

```
pip install -e .
```

End with an example of getting some data out of the system or using it for a little demo. Run the notebooks:

```
jupyter notebook
```

🎈 Usage
After installation, the primary way to use and understand PY-GREX is by exploring the Jupyter Notebooks included in the repository. These notebooks demonstrate how to apply the various explainability methods and evaluation metrics to sample datasets.


## ⛏️ Built Using <a name = "-built-using-"></a>
Python - Core Language (v3.11)

PyTorch - Deep Learning & Tensor Computation

Conda - Environment & Package Management

Jupyter Notebook - Examples & Demonstrations

pip - Package Installer

## ✍️ Authors <a name = "-authors-"></a>

@mateustoledo - py-grex creator and repository owner
@ludovikcoba - Initial work on recoxplainer