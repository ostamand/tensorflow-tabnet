# Tested TensorFlow TabNet

A TensorFlow 2.X implementation of the paper [TabNet](https://arxiv.org/abs/1908.07442).

TabNet is already available for TF2 [here](https://github.com/titu1994/tf-TabNet). However, an error in the batch normalizations of the shared feature transformer blocks makes it unable to learn. In the original code, only the fully connected layers weights are shared, not the batch normalizations which remain unique. When sharing the batch normalizations, training of TabNet on the datasets & hyperparameters of the original paper yields extremely bad results.

Moreover, the implementation uses `virtual_batch_size` from `tf.keras.layers.BatchNormalization` which has a couple of issues:
- Does not allow batch size which can't be divided by the selected virtual batch size used during training (even for inference which should be independent of batch size).
- Less update of the batch normalizations wrt to Ghost Batch Normalization.

Below is a plot of the accuracy for a model trained with a correct Ghost Batch Normalization and one trained with the incorrect `virtual_batch_size`:

![GitHub Logo](images/virtual_bs_vs_gbn.png)


Probably not aware of the issues introduced above, the implementation proposes to use [Group Normalization](https://arxiv.org/abs/1803.08494) instead of [Ghost Batch Normalization](https://arxiv.org/abs/1705.08741) which does make things better and able to learn. However, no comparision of the results obtained is proposed. Are they even close to the original ones? Can the model really generalize and also obtain state of the art results?

Therefore, a new correct and tested TF2 implementation of TabNet is proposed. It tries to not simply port the original code but take advantage of TF2 modular approach to make it easier to finetune and train a TabNet model on different tasks.

## Setup

### For development

```bash
python -m venv venv
source venv/bin/activate
chmod +x scripts/install.sh; ./scripts/install.sh
```

## Dataset
- http://archive.ics.uci.edu/ml//datasets/Covertype
- https://www.kaggle.com/uciml/forest-cover-type-dataset/data#

## Reference
- [paper](https://arxiv.org/abs/1908.07442)
- [tf-TabNet TF2.0](https://github.com/titu1994/tf-TabNet)
- [original code base](https://github.com/google-research/google-research/tree/master/tabnet)