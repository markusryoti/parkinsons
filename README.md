# Parkinsons classifier

Simple classifier for recognizing Parkinsons disease. Data downloaded from https://www.kaggle.com/datasets/kmader/parkinsons-drawings/code

## Current state

Accuracy with _Resnet50_ based model ~85 %

### Example predictions

![Predictions](results/predictions.png 'a title')

### Training losses

![Losses](results/losses.png 'a title')

## TODO

- [ ] Split training and testing data better - should always see similar number of spirals and waves, healthy and not healthy ones
- [ ] Code cleanup
- [ ] Deploy
