# M-SpeechCLIP
Implementation of the M-SpeechCLIP model, introduced in the paper ["M-SpeechCLIP: Leveraging Large-Scale, Pre-Trained Models for Multilingual Speech to Image Retrieval"](https://arxiv.org/abs/2211.01180)

## Environment Set-Up
The primary requirements for this project are [PyTorch](https://pytorch.org/get-started/locally/) and [CLIP](https://github.com/openai/CLIP).
You can get the exact versions we used by calling:

`pip install -r requirements.txt`

## Data Preparation
The following files are needed:
- Metadata json files for each split (train, valid, test) for each language you will train or test on
- Either:
  - A directory containing all of the images in the dataset, OR
  - A pickle file containing the pre-computed CLIP embedding of each image
- Directories containing the caption wavs for each language you will train or test on
- A json file containing a dictionary whose keys are image filenames and whose paths are English captions (optional, allows you to evaluate zero-shot text-speech retrieval)

You can specify where each of these is located by modifying `data_paths.py`

## Model Training
All model variants can be trained using `main.py`. Hyperparameters are architecture variants are controlled by the command line arguments defined in `args.py`, and can be examined by running:

`python3 main.py --help`

Remember to set `CUDA_VISIBLE_DEVICES=` to the set of GPUs you'll be training on, and to specify the number of GPUs if it differs from the default of 2. You'll also likely want to specify a non-default checkpoint to save your model at. For example, say you want to train an M-SpeechCLIP model on the full tri-lingual Places100 dataset with cross-lingual loss terms, but you don't want to unfreeze the HuBERT feature extractor, and you want to use the Large variant of HuBERT. You want to train the model for 50 epochs, and display the current training loss every 200 training steps. The GPUs available to you are 2, 3, 5, and 7, and you want to save the model at `~/M-SpeechCLIP/frozen_extractor_with_XLL.pth`. Then you would run:

`CUDA_VISIBLE_DEVICES=2,3,5,7 python3 main.py --dataset=PlacesMulti --loss_type=CrossLingual --gpus=4 --epochs=50 --display=200 --chkpt_path=~/M-SpeechCLIP/frozen_extractor_with_XLL.pth`

## Model Testing
The easiest way to test a model is to take the command you used to train it, and modify the `epochs` argument to 0. This will output Recall@1, 2, 5, and 10 on both the validation and testing set that correspond to the data you trained on. If you trained using multilingual batches, and want to evaluate performance on monolingual batches, you can pass the argument `monolingual_batches=1`, and you'll get retrieval scores for each language separately.

For testing transfer to the tasks of speech-text and cross-lingual speech-speech retrieval, you can modify the script `test_zero_shot_transfer.py` to instantiate the model you want to test, and then run it to output all possible retrieval directions between English Text, English Speech, Hindi Speech, Japanese Speech, and Images.
