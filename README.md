# "Are Two Heads the Same as One? Identifying Disparate Treatment in Fair Neural Networks".

Welcome. This is the code release of our 2022 NeurIPS paper [Are Two Heads the Same as One? Identifying Disparate Treatment in Fair Neural Networks](https://arxiv.org/abs/2204.04440).

## Examples
Please read the [overview_notebook.ipynb](overview_notebook.ipynb) with 
examples, results, and explanations. It takes some time to run all models, but 
if you want to run all models yourself, run `all_jobs.sh` after the following setup.


## Setup

1) Download img_align_celeba.zip and list_attr_celeba.txt from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
and move it to data/celeba/.

2) Create a virtual environment with python==3.8 and run 
````
python -m pip install -r requirements.txt
````

3) Install torch and torchvision. We used torch==1.12.1+cu116 and torchvision==0.13.1+cu116.

torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116



## Citing this paper

If you use this software or want to refer to it, please cite the following publication:
```
@inproceedings{lohaus2022,
  title={Are Two Heads the Same as One? Identifying Disparate Treatment in Fair Neural Networks},
  author={Lohaus, Michael and Kleindessner, Matth{\"a}us and Kenthapadi, Krishnaram and Locatello, Francesco and Russell, Chris},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```