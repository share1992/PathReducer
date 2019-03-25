#PathReducer 2.0

## Installation
`pip install git+https://github.com/share1992/dimensionality_reduction_PCA.git@restructure` to install.

## Read xyz file
```
from pathreducer.filereaders import XYZReader

data = XYZReader('filename')
print(data.coordinates.shape)
print(data.elements.shape)
```
