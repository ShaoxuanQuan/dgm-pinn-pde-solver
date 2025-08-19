# DGM-PINN Solver for Financial PDEs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning solver for financial Partial Differential Equations (PDEs), including the 2D Black-Scholes and Hamilton-Jacobi-Bellman (HJB) equations. The model is based on the Deep Galerkin Method (DGM) and incorporates enhancements for improved performance and faster convergence.

## Core Approach

The solver's architecture is built on the following key concepts:

- **Deep Galerkin Method (DGM):** The core framework is based on the DGM algorithm by Sirignano & Spiliopoulos (2018), which uses a neural network(lstm) to approximate the PDE solution.The model also incorporates common strategies from gPINN to better enforce physical constraints and improve convergence.

- **Controlled Drift Sampler:** A key feature is the integration of the controlled drift sampler. This approach is heavily inspired by the work of and focuses sampling on more critical regions of the problem domain,which are generated from SDE routine of corresponding PDE .

## References and Acknowledgments

This work is inspired by and adapts concepts from the following research:

- **Primary Inspiration:** Kelbel, F. (2024). [*Optimal Control of Agent-Based Dynamics under Deep Galerkin Feedback Laws*](https.github.com/FreditorK/Optimal-Control-of-Agent-Based-Dynamics). The controlled drift sampler is a key component adapted from this work.

- **Foundational Algorithm:** Sirignano, J., & Spiliopoulos, K. (2018). DGM: A deep learning algorithm for solving partial differential equations. *Journal of Computational Physics*.

## License

This project is licensed under the MIT License
