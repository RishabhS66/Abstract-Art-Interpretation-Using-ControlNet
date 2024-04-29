# Controlling-Stable-Diffusion
ControlNet training with custom control images

The dataset was created using the URLs in the [1% data sample file](https://storage.googleapis.com/gresearch/wit/wit_v1.train.all-1percent_sample.tsv.gz) of the [Wikipedia-based Image Text (WIT)](https://github.com/google-research-datasets/wit/blob/main/DATA.md) dataset. Download the zip file, unzip it, and rename the file obtained to `data.tsv` before running the Python notebook `create_dataset.ipynb` to generate the dataset.

The training dataset can be found [here](https://www.kaggle.com/datasets/rishabhsrivastava66/images-made-up-of-geometric-shapes-controlnet/data). Make sure the folder structure looks like this - 

    training
    ├── geometricShapes14k
        ├── prompt.json
        ├── source
        ├── target
    
