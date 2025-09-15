<p align="center">
  <a href="https://github.com/toledomateus/py-grex" rel="noopener">
  <img width=200px height=200px src="https://raw.githubusercontent.com/toledomateus/py-grex/main/assets/pygrex-logo.png" alt="Project logo"></a>
</p>

<h3 align="center">PY-GREX: An Explainable Group Recommender Systems Toolkit</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)](https://github.com/toledomateus/py-grex) 
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/toledomateus/py-grex.svg)](https://github.com/toledomateus/py-grex/issues) 
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/toledomateus/py-grex.svg)](https://github.com/toledomateus/py-grex/pulls) 

</div>

---

<p align="center"> A software toolkit for explainable group recommender systems, including several state-of-the-art explainability methods and evaluation metrics.
    <br> 
</p>

## ✨ Live Demo

Experience PY-GREX without any installation through our interactive web application, built with Streamlit.

**➡️ [Launch the App: pygrex.streamlit.app](https://pygrex.streamlit.app/)**

![Live Demo of the PY-GREX App](assets/pygrex-video-demo.gif)

---

## 📝 Table of Contents
- [🧐 About](#-about)
- [🚀 Features](#-features)
- [🏁 Getting Started](#-getting-started)
- [🎈 Usage](#-usage)
- [🤝 Contributing](#-contributing)
- [⛏️ Built Using](#️-built-using)
- [✍️ Authors](#️-authors)
- [📜 License](#-license)

---

## 🧐 About

Recommender systems heavily shape our digital experiences. Consequently, there's a growing demand for insight into how these systems generate predictions, not just for individuals but also for groups of users.

Recognizing that explanations enhance trust, efficiency, and even persuasive power, researchers have actively pursued this area. Yet, the field lacks a standard, accessible toolkit for implementing and evaluating these techniques, especially for the nuanced domain of group settings.

PY-GREX addresses this critical need, offering a modular Python toolkit equipped with multiple state-of-the-art explainability algorithms to facilitate research and development in eXplainable AI (XAI) for Recommender Systems.

---

## 🚀 Features

PY-GREX provides a modular, end-to-end pipeline for explainable group recommendations.

- **Recommendation Models**:
  - Matrix Factorization (ALS, BPR)
  - Neural Collaborative Filtering (GMF, MLP)
  - Explainable Models (Autoencoder, EMF)

- **Group Aggregation Strategies**:
  - Consensus-Based (Additive, Multiplicative, Average)
  - Majority-Based (Borda Count)
  - Borderline (Least Misery, Most Pleasure, Most Respected Person)

- **Explanation Methods**:
  - **Counterfactual**: Sliding Window Explainer
  - **Rule-Based**: EXPGRS (Association Rules Explainer)
  - **Local Explainers**: LORE4Groups *(planned)*

- **Evaluation Metrics**:
  - **Accuracy**: Hit Ratio (HR), nDCG
  - **Explainability**: Model Fidelity, Feature Coverage Ratio

---

## 🏁 Getting Started

### For Users (The Easy Way)
The best way to get started is to use the live Streamlit application. No installation is required!

➡️ **[Explore the PY-GREX App](https://pygrex.streamlit.app/)**

### For Developers (Local Setup)
If you want to run the project locally for development or to use the library programmatically:

1.  **Prerequisites**: Make sure you have Conda and Git installed.

2.  **Clone the repository**:
    ```bash
    git clone [https://github.com/toledomateus/pygrex.git](https://github.com/toledomateus/pygrex.git)
    cd pygrex
    ```

3.  **Create and activate a Conda environment**:
    ```bash
    # PY-GREX was developed with Python 3.11
    conda create -n pygrex python=3.11
    conda activate pygrex
    ```

4.  **Install dependencies**: Install all required packages using the `requirements.txt` file. This includes PyTorch, Streamlit, and all other necessary libraries.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the local Streamlit app**:
    ```bash
    streamlit run Home.py
    ```

---

## 🎈 Usage

### Interactive Web App
The primary way to use PY-GREX is through the web application. It allows you to:
-   **Upload or use default data** for users, items, and groups.
-   **Select and train** a variety of recommendation models.
-   **Generate group recommendations** using different aggregation strategies.
-   **Produce and evaluate explanations** for the recommendations using state-of-the-art methods.

### Programmatic Use (Jupyter Notebooks)
For a deeper dive into the library's functions, explore the Jupyter Notebooks included in the repository. These demonstrate how to apply the various explainability methods and evaluation metrics directly in your own code.

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## ⛏️ Built Using
- **Python** - Core Language (v3.11)
- **Streamlit** - Web Application Framework
- **PyTorch** - Deep Learning & Tensor Computation
- **Pandas** & **NumPy** - Data Manipulation
- **Conda** - Environment Management
- **mlxtend** - Association Rule Mining

---

## ✍️ Authors
- **@toledomateus** - Project Creator

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.