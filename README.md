# MLNanoShaper

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://concept-lab.github.io/MLNanoShaper.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://concept-lab.github.io/MLNanoShaper.jl/dev/)
[![Build Status](https://github.com/concept-lab/MLNanoShaper.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/hack-hard/MLNanoShaper.jl/actions/workflows/CI.yml?query=branch%3Amain)

MLnanoshaper is an experimental sofware designed to compute proteins surface like the Nanoshaper program but with less computations using Machine Learning.
Instead of having a probe (a virtual water molecule) that rolls over the surface, like in Nanoshaper, we will have a machine learning algorithme will generate a scalar "energy" field from the atoms. We will then define the surface as an implicit surface ie $\{x/ f(x)= 0\}$ that will be triangulated using marching cubes.
