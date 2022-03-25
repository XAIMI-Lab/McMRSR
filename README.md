### Data Description
Two in-house private datasets: pelvic and brain and one public dataset: fastMRI are utilized in our experiments. For the private data, all studies have been approved by the local Institutional Review Board (IRB). The IRB asked us to protect the privacy of participants and to maintain the confidentiality of data. Since we cannot make the two datasets publicly available, we won't claim them as our contribution.
### Environment and Dependencies
Requirements:
* Python 3.6
* Pytorch 1.4.0 
* scipy
* scikit-image
* opencv-python
* tqdm

Our code has been tested with Python 3.6, Pytorch 1.4.0, torchvision 0.5.0, CUDA 10.0 on Ubuntu 18.04.


### To Run Our Code
- Train the model
```bash
python train_demo.py
```

- Test the model
```bash
python test_demo.py --resume ' '
```
where
`--resume`  trained model. 

