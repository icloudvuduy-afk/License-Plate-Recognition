# Vietnamese License Plate Recognition

## Installation

```bash
  git clone https://github.com/icloudvuduy-afk/License-Plate-Recognition.git
  cd License-Plate-Recognition

  # install dependencies using pip 
  pip install -r ./requirement.txt
```

## Run License Plate Recognition

```bash
  # run inference on webcam (15-20fps if there is 1 license plate in scene)
  python webcam.py 

  # run inference on image
  python lp_image.py -i test_image/3.jpg
