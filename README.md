# Controlling-Stable-Diffusion
ControlNet training with custom control images

### Training Dataset

The training dataset can be found [here](https://www.kaggle.com/datasets/rishabhsrivastava66/images-made-up-of-geometric-shapes-controlnet/data). Make sure the folder structure looks like this - 

    training
    ├── geometricShapes14k
        ├── prompt.json
        ├── source
        ├── target

The training dataset was created using the URLs in the [1% data sample file](https://storage.googleapis.com/gresearch/wit/wit_v1.train.all-1percent_sample.tsv.gz) of the [Wikipedia-based Image Text (WIT)](https://github.com/google-research-datasets/wit/blob/main/DATA.md) dataset. Download the data sample zip file, unzip it, and rename the file obtained to `data.tsv` before running the Python notebook `create_dataset.ipynb` to generate the dataset. You need to install [Primitive](https://github.com/fogleman/primitive) in your command line if you want to generate a similar dataset for yourself or generate the control image for an image of your own. Primitive helps you convert your images to abstract images using geometric primitives like triangles.

Images downloaded using the URLs are used to generate the control images for the ControlNet. The control images are made up of triangles. 
