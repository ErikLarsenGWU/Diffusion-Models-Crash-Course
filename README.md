# Diffusion Models Crash Course

**A hands-on, graduate-level introduction to diffusion and flow-based generative modeling using TensorFlow 2.x and Keras.**

This repository documents a structured course in the mathematics, implementation, and evaluation of generative models. The approach begins with simple two-dimensional probability distributions, builds denoising diffusion probabilistic models (DDPMs) from first principles, and progresses toward conditional flow matching, neural ordinary differential equations, and image generation.

The emphasis is on **understanding the derivations, implementing the algorithms, and testing whether the generated distribution actually matches the data**—not just producing attractive samples.

## Learning objectives

By working through the course, learners will:

- Develop intuition for Gaussian distributions, Gaussian mixtures, covariance, and KL divergence.
- Derive and implement the DDPM forward-noising process and reverse denoising process.
- Understand why a closed-form forward diffusion conditional enables efficient training.
- Train time-conditioned neural networks to predict noise or velocity fields.
- Explore conditional flow matching and neural ODE-based generation.
- Compare diffusion and flow-based models on common datasets and evaluation criteria.
- Progress from synthetic two-dimensional data to image-generation experiments.

## Course roadmap

| Module | Topic | Focus |
| --- | --- | --- |
| **1** | Probability and Gaussian mixtures | Gaussian density, covariance, multimodal data, KL divergence, ODE/SDE foundations |
| **2** | Denoising Diffusion Probabilistic Models | Forward process, noise schedules, noise-prediction objective, reverse sampling |
| **3** | Conditional flow matching and neural ODEs | Probability paths, velocity fields, ODE integration, generative sampling |
| **4** | Diffusion vs. flow matching | Controlled comparison of training, sampling, and generated distributions |
| **Beyond the core** | Advanced generative modeling | MNIST, score-based SDEs, rectified flows, guidance, and evaluation |

Modules are developed progressively. The roadmap describes planned coverage; not every module is necessarily available in the repository yet.

## How to use this repository

Each module is intended to have two complementary learning artifacts:

1. **Lesson PDF:** Theory, mathematical derivations, examples, references, exercises, and solutions or solution guidance.
2. **Student Jupyter notebook (`.ipynb`):** A Markdown-only learning template containing explanations, equations, prompts, and unanswered exercises. Add your own code cells to complete the lab.

A recommended workflow is to read the lesson, derive the equations by hand, complete the notebook independently, and then compare the results against the lesson material. Record your assumptions, plots, quantitative metrics, and interpretations alongside your implementations.

> **Repository note:** File and directory names may evolve as modules are added. Use the repository's file browser to locate the latest PDFs and notebooks.

## Mathematical preview: DDPM

The forward diffusion transition adds a controlled amount of Gaussian noise:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I\right).
$$

With $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$, we can sample a noisy observation directly at any timestep:

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I).
$$

This **closed-form forward sampling equation** avoids simulating all preceding noise steps during training. A neural network learns to predict the noise from $x_t$ and the timestep $t$; a reverse-time procedure then generates new observations.

## Technology stack

- **Python 3**
- **TensorFlow 2.x and Keras** for neural networks and model training
- **NumPy** for numerical analysis
- **Matplotlib** for visualization
- **Jupyter Notebook / JupyterLab** for interactive exercises

**Framework policy:** Course implementations use TensorFlow/Keras, not PyTorch.

### Suggested local setup

Create and activate a Python virtual environment, then install the core packages:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install tensorflow numpy matplotlib jupyter
jupyter lab
```

GPU support and installation requirements depend on your operating system, hardware, and TensorFlow version. Consult the [official TensorFlow installation guide](https://www.tensorflow.org/install) for platform-specific instructions.

## Evaluating generative models

Visualizations are useful but not sufficient. The course emphasizes quantitative checks such as:

- Empirical means and covariance matrices compared with target values.
- Mode coverage and relative occupancy for multimodal distributions.
- Distance to known component centers and other distributional diagnostics.
- Training loss, sampling speed, and sensitivity to hyperparameters.

In particular, matching the mean and covariance does **not** guarantee that a model has captured the modes of a Gaussian mixture.

## Open learning resources

- [Ho, Jain & Abbeel — *Denoising Diffusion Probabilistic Models* (2020)](https://arxiv.org/abs/2006.11239)
- [Song et al. — *Score-Based Generative Modeling through Stochastic Differential Equations* (2021)](https://arxiv.org/abs/2011.13456)
- [Prince — *Understanding Deep Learning* (free textbook)](https://udlbook.github.io/udlbook/)
- [Murphy — *Probabilistic Machine Learning* (free textbooks)](https://probml.github.io/pml-book/)
- [MIT — Flow and Diffusion Models](https://diffusion.csail.mit.edu/)
- [Keras DDPM example](https://keras.io/examples/generative/ddpm/)

## Project status

This is an evolving educational repository. Modules, notebooks, exercises, and experimental results will be added and refined over time. The work is designed for reproducible study and exploration, rather than as a production-ready generative modeling library.

## Author

**Erik Larsen**  
[GitHub profile](https://github.com/ErikLarsenGWU)

## License

No license is specified here. Check the repository for a `LICENSE` file before assuming reuse permissions.
