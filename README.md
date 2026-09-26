# Cat vs. Dog Classifier
A CNN that I trained from scratch in PyTorch to tell if a photo shows a cat or a dog.
## Data
The photos come from the Kaggle "Dogs vs. Cats" dataset. I use this copy of it from GitHub, which has about 25,000 photos: [laxmimerit/dog-cat-full-dataset](https://github.com/laxmimerit/dog-cat-full-dataset).
`src/prepare_data.py` downloads it and picks a random subset:
- train: 2000 cats and 2000 dogs
- val: 500 cats and 500 dogs, from the same folder as train but different photos
- test: 500 cats and 500 dogs, from the dataset's separate test folder

The test photos are not used during training at all, only at the end in `src/evaluate.py`.
## Model
The model is in `src/model.py`. It has 4 convolutional blocks with 32, 64, 128 and 256 channels. Each block has two 3x3 convolutions with batch norm and ReLU, and then max pooling. After the blocks there is average pooling and two linear layers with dropout.
All photos are resized to 64x64. The original photos are much bigger, but training on them at full size would take too long on a CPU.
For training I use Adam with a learning rate that goes down over the epochs. The training photos also get random crops and flips.
## Results
I trained for 25 epochs, which took about an hour and a half on my laptop's CPU. The best validation accuracy was 88.5% at epoch 24, and that is the model saved in `outputs/checkpoints/best_model.pt`.
On the 1000 test photos the model gets 90.0% accuracy:

```
              precision    recall  f1-score   support

         cat      0.900     0.900     0.900       500
         dog      0.900     0.900     0.900       500

    accuracy                          0.900      1000
   macro avg      0.900     0.900     0.900      1000
weighted avg      0.900     0.900     0.900      1000
```
![Training curves](outputs/figures/training_curves.png)

Training accuracy goes up to about 92%, while validation accuracy stays around 88%. So the model overfits a little, which is expected for a small network trained from scratch on 4000 photos.

![Confusion matrix](outputs/figures/confusion_matrix.png)

50 of the 500 cats were classified as dogs and 50 of the 500 dogs as cats, so the mistakes are split evenly between the two classes.

![Sample predictions](outputs/figures/sample_predictions.png)
## How to run
You need Python 3.10 or newer.

```
pip install -r requirements.txt
python src/prepare_data.py
python src/train.py
python src/evaluate.py
python src/predict.py examples/cat_example.jpg
```
`prepare_data.py` creates the `data/` folder, `train.py` trains the model, `evaluate.py` tests it and saves the figures, and `predict.py` shows the prediction for one photo. You can also try `predict.py` on your own photos, but the model only knows cats and dogs, so any other picture will still be labeled as one of them.
