# RecStudio

[![PyPi Latest Release](https://img.shields.io/pypi/v/recbole)](https://pypi.org/project/recbole/)
[![Conda Latest Release](https://anaconda.org/aibox/recbole/badges/version.svg)](https://anaconda.org/aibox/recbole)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-RecBole-%23B21B1B)](https://arxiv.org/abs/2011.01731)

RecStudio is an efficient, unified and comprehensive recommendation library based on PyTorch. All the algorithms can be 
divided into the following four categories according to the different tasks.

- General Recommendation
- Sequential Recommendation
- Knowledge-based Recommendation
- Social-Network-based Recommendation

At the core of the library, we divide all the models into 3 basic classes according to the number of
towers:

- `TowerFreeRecommender`: 