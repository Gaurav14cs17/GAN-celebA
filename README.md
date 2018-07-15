# GAN-celebA
Tensorflow implementation of GAN in Dataset CelebA.
## Prerequisites

- Python 3.3+
- [Tensorflow 1.4.0](https://www.tensorflow.org/)
- [SciPy](http://www.scipy.org/install.html)
- [pillow](https://github.com/python-pillow/Pillow)

## Usage
To train a model with dataset CelebA:
 ```
 $ python main.py
 ```
To test with an existing model:
 ```
 $ python main.py --train=False
 ```

## Results


## Folder structure
The following shows basic folder structure.
```
├── main.py # gateway
├── data
│   ├── celebA # celebA data (not included in this repo)
│       ├── xxxx.jpg
│       ├── xxxx.jpg
│       ├── xxxx.jpg
│       └── xxxx.jpg
├── GAN.py # build GAN
├── model.py # Generator and Discriminator
├── ops.py # some operations on layer
├── utils.py # utils
├── logs # log files for tensorboard to be saved here
└── checkpoint # model files to be saved here
```

