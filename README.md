# Human Protein Atlas Image Classification

This project is my attempt at the [Human Protein Atlas Classification](https://www.kaggle.com/c/human-protein-atlas-image-classification) competition on Kaggle.
To make the model work better, I've used multiple approaches and optimizations such as:
1. Multilabel stratification
2. 4 channel Resnets
3. Making transfer learning work for 4 channels
4. Data Augmentation
5. One Cycle Learning rate scheduling (Cosine Annealing)
6. Progressive resizing
7. Discriminative layer training

At the time of taking part in this competition, I had to suddenly stop working on this project due to some unforeseen circumstances. As a result I couldn't train the model for a significant amount of time.
I managed to get an accuracy of about **88%** on the multilabel classification task invloving 29 classes.
The best fbeta score was **0.412**.

> This is an incomplete / dropped project
