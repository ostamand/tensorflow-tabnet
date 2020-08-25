# Benchmarks

To make sure the implementation of TabNet is correct, a series of benchmarks on different datasets are investigated and compared to the original paper results. Most of the time the exact same hyperparameters are used (indicated when not the case). Some differences are expected since dataset splits are not the same.

## Covertype

Classification of forest cover type from carthographic variables.

To run the training:

```python
python3 benchmarks/covertype.py
```

The following hyperparameters are used by default (can easily be changed thru arguments to the `covertype.py` script)

| Hyperparameter | Value |
| -------------- | ----- |
| Feature dim.   | 64    |
| Output dim.    | 64    |
| Sparsity Coeff.| 0.0001|
| Batch Size     | 163284|
| Virtual batch size | 512 |
| Batch Norm. Momentum | 0.7 |
| Number of steps | 5 |
| Relaxation factor | 1.5 |
| Minimum learning rate  | 1e-6|
| Decay steps | 500 |
| Total steps | X (was 130000) |
| Clip norm. | 2.0 |
| Dropout rate | 0.2 |


Here is a summary of the changes with respect to the paper implementation

- Add warmup which helps generalization when a large batch size is used [(reference)](https://arxiv.org/abs/1906.03548)
- Add dropout on the classifier head
- Add an option to used infernce example weighing by providing a `alpha > 0` during inference [(reference)](https://arxiv.org/abs/1906.03548)

To optimize the inference weighting `alpha` on the validation dataset use:

```python
python3 benchmarks/covertype_opt_inf.py --model_dir .outs/w200
```

where `--model_dir` is where the model was saved after training.

Results obtained are summarized in the table below: