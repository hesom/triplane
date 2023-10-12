# nerfstudio-triplane
Template repository for creating and registering methods in Nerfstudio.

## Registering with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/quickstart/installation.html). Clone or fork this repository and run the commands:

```
conda activate nerfstudio
cd triplane
pip install -e .
ns-install-cli
```

## Running the new method
This repository creates a new Nerfstudio method named "triplane". To train with it, run the command:
```
ns-train triplane --pipeline.model.mip-levels 3 --pipeline.model.mip-method laplace --data [PATH TO BLENDER DATA]
```
