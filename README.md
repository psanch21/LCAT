

# Learnable Graph Convolutional Attention Networks (LCAT)

This is the official Pytorch implementation of CAT and L-CAT.



## Installation

For CPU usage:


```yaml
conda create --name lcat python=3.9.7  --no-default-packages
conda activate lcat

pip install torch==1.13.1 torchvision==0.14.1


pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html --force-reinstall

```

For GPU usage:

```bash
conda create --name lcat_gpu python=3.9.7 --no-default-packages
conda activate l

pip3 install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html 
```

Then install the following dependencies:

```bash
pip install pytest==7.2.0
pip install pytest-helpers-namespace
```



## Usage

To use the L-CAT in your own project, see the `tests` folder for examples.
In particular, check out `tests/test_lcat_pyg.py` to learn how to use LCAT with GCN, GCN2, and PNA with the PyTorch Geometric implementation.


## Testing

This project uses pytest for testing. To run the tests, you can use the following command:

```bash
pytest tests
```


This will run all the tests in the `tests/` directory. You can also run individual test files using the following command:

```bash
pytest tests/test_lcat_pyg.py
```

# License

This project is licensed under the MIT License.


## Citing

If you use CAT or L-CAT, consider citing our paper

```bibtex
@inproceedings{javaloy2023learnable,
title={Learnable Graph Convolutional Attention Networks},
author={Adri{\'a}n Javaloy and Pablo Sanchez Martin and Amit Levi and Isabel Valera},
booktitle={International Conference on Learning Representations (ICLR) },
year={2023},
url={https://openreview.net/forum?id=WsUMeHPo-2}
}
```



## Contact

For further information: <a href="mailto:psanchez@tue.mpg.de">psanchez@tue.mpg.de</a>







