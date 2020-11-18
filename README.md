# An MTO-based DNN Training Algorithm with the Auxiliary Tasks Formulated From the Aspect of Loss Functions
The Pytorch source code of Chapter 4 in thesis [Training Deep Neural Networks via Multi-task Optimisation]
If you use this code, please cite the thesis.


****
# Usage

## Download data
First, you need to download the following datasets and put them in the directory "data/".
1. ipiu: [Google Drive](https://drive.google.com/open?id=15ZkeUTMmrUiuTM2IzbHlQobEFxI8XNV2), [Baidu NetDisk](https://pan.baidu.com/s/1Ouc94jWDxohd_2VLom2ClQ)(password: izwf)
2. lfw: [Google Drive](https://drive.google.com/open?id=1jVqSrYfnIcRu1pPGYLtAIjYQtFrLJ1fJ), [Baidu NetDisk](https://pan.baidu.com/s/1sWdI-X7TOXcSBPCxdX1tBg)(password: 6afw)
3. rsscn7: [Google Drive](https://drive.google.com/open?id=1xYdiqUqU7olVQeh7GZuN9OPonEEW70nU), [Baidu NetDisk](https://pan.baidu.com/s/1GpfPu1Chf9VXJs1brhjSSA)(password: u5m9)

After unzip, each dataset folder contains a "train" and a "val" image folder.

## Training
1. Use the following code for MTO based training (e.g., on ipiu dataset),
```
python main.py -train_dir ipiu -cuda -alpha 1.0
```

2. Use the following code for conventional training,
```
python main.py -train_dir ipiu -cuda -alpha 0.0
```

3. Use the following code for baseline of combined loss,
```
python main_combineloss.py -train_dir ipiu -cuda
```

4. Use the following code for baseline of deep MTL,
```
python main_mtl.py -train_dir ipiu -cuda
```

After training, the generated models are stored in folder "results".

## Evaluation
Use the following code for evaluation on the vdsr of first run (e.g., on ipiu dataset),
```
python evaluate.py -cuda -test_set ipiu -run 0 -model vdsr8x1e-03
```
Modify the parameter -run to evaluate the result of the specific run.

## To generate HR images
Use the following code to generate HR images,
```
python super_resolve.py -scale [scale factor: int] -hr_set [path/to/the/HR_images]
```
Modify other parameters in the head of the code for specific configuration. 
