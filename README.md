# facematch
1 module to compare 2 faces in Python 3
```
git clone https://github.com/ma9o/facematch
cd facematch
pip3 install dist/facematch-1-py3-none-any.whl
```

To emulate a production environment (and speed up testing) the face recognition model runs on a Docker container
```
docker pull ma9o/facematch
docker run --rm -it -p 50052:50052/tcp ma9o/facematch
```

## Usage
```python3
from facematch import FaceMatch
affinity = FaceMatch('image1.png','image2.png').compare()
```

## Credits
* https://arxiv.org/pdf/1804.06655.pdf
* https://github.com/deepinsight/insightface
* https://github.com/tomerfiliba/rpyc
* https://github.com/dmlc/tvm
