# perCLTV

This repo is the TF2.0 implementation of perCLTV: A General System for Personalized Customer Lifetime Value Prediction in Online Games [[PDF](https://dl.acm.org/doi/abs/10.1145/3530012)].

## Folders

- `data/`: data of perCLTV (**randomly generated sample data** to show the data format, **not the real data**).
  - `sample_data_individual_behavior.csv`: the sample for individual behavior sequential data.
  - `sample_data_social_behavior.csv`: the sample for social behavior graph data.
  - `sample_data_label.csv`: the sample data for label, where label1 is churn label (binary classification) and label2 is payment label (regression).
- `src/`: implementations of MSDMT.
  - `model.py`: the code for model.
- `main.py`: the code for pipeline.

## Requirements

The code has been tested running under Python 3.8.16, with the following packages installed (along with their dependencies):

- tensorflow == 2.12.0
- spektral == 1.2.0
- attention == 5.0.0
- numpy == 1.23.5
- pandas == 2.0.0
- scikit-learn == 1.2.2

**Note**: to fixup "multiple 'training' args" bug in attention, open attention.py and modify the following code:

```python
  class Attention(Layer):
    
    # ...

    def __call__(self, inputs, training=None, **kwargs):
        if debug_flag:
            # return self.call(inputs, training, **kwargs)
            return self.call(inputs, training=training, **kwargs)
        else:
            # return super(Attention, self).__call__(inputs, training, **kwargs)
            return super(Attention, self).__call__(inputs, training=training, **kwargs)

```

## Running

```shell
python main.py 
```

## Cite

Please cite our paper if you use this code in your own work:

```text
@article{zhao2023percltv,
  title={perCLTV: A general system for personalized customer lifetime value prediction in online games},
  author={Zhao, Shiwei and Wu, Runze and Tao, Jianrong and Qu, Manhu and Zhao, Minghao and Fan, Changjie and Zhao, Hongke},
  journal={ACM Transactions on Information Systems},
  volume={41},
  number={1},
  pages={1--29},
  year={2023},
  publisher={ACM New York, NY}
}
```
