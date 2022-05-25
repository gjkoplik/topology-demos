#!/bin/bash
conda env create -f topology_demos_env.yml
source activate topology_demos_env
python -m ipykernel install --user --name topology_demos_env --display-name "Python (topology_demos_env)"