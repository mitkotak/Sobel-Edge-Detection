# Sobel-Edge-Detection
Implemented sobel edge detection using Cuda

Fire up your terminal/command prompt, navigate to the Github folder

```sh
make
```
Note : We assume that `python3` is installed on your machine

#### Validation test

```sh
python3 test/sobel_validation.py
```
Note : We assume that you have run `make` atleast once

#### Test Results 

All of the test results are stored as `test/benchmarks_graph.csv` and `test/benchmarks_non_graph.csv`

#### Changing the Image 

``
pgmread("images/coins.ascii.pgm", (void *)image, width, height);
``
to 
``
pgmread("images/..", (void *)image, width, height);
``
Make sure you change `width` and `height` as well
