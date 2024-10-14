# Simple and Minimalistic Implementation of VQGAN

Paper: https://arxiv.org/abs/2012.09841

Reference Github Code: https://github.com/CompVis/taming-transformers/tree/master

### Download Open Images

I used AWSCLI to download the dataset. Reference: https://github.com/cvdfoundation/open-images-dataset?tab=readme-ov-file#download-images

create a dataset folder using:

```
mkdir open-images-dataset
cd open-images-dataset
```

Next:
- Install AWSCLI: https://aws.amazon.com/cli/
- Download Train set using: `aws s3 --no-sign-request sync s3://open-images-dataset/train` This will take 513GB
- Download Test set using: `aws s3 --no-sign-request sync s3://open-images-dataset/test`  This will take 36GB

### Start training

Once you are done downloading dataset, now change L45 and L49 inside `main.py`. The will point to the train/validation/test set folders of open-images.

Change L36 in `main.py` for your WANDB.

Start training:

```python main.py```

