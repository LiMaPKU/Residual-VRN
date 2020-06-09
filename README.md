# Residual-VRN
 L. Ma, Y. Tian, P. Xing and T. Huang, "Residual-Based Post-Processing for HEVC," in IEEE MultiMedia, vol. 26, no. 4, pp. 67-79, 1 Oct.-Dec. 2019

![](Residual-Based%20Post-processing%20for%20HEVC.jpg)

**The result are as follows:**

| Sequences       | Residual-VRN  | DR-16 | DR-32 | DR-64 |
| ---             | :------:      | :---: | :---: | :---: |  
|Kimono           |         -5.2% | -5.4% |  -5.9%|-8.0%  |
|ParkScene        |         -5.6% | -6.6% |  -6.5%|-8.9%  |
|Cactus           |         -6.3% | -6.9% |  -7.2%|-9.4%  |
|BasketballDrive  |         -5.9% | -6.4% |  -7.1%|-9.7%  |
|BQTerrace        |         -4.0% | -4.4% |  -4.6%|-5.7%  |
|BasketballDrill  |        -11.0% | -12.9%| -13.9%|-15.8% |
|BQMall           |         -8.3% | -9.3% |  -9.8%|-11.0% |
|PartyScene       |         -5.5% | -6.1% |  -6.3%|-7.0%  |
|RaceHorses       |         -5.2% | -5.5% |  -5.7%|-8.9%  |
|BasketballPass   |         -9.3% | -10.6%| -11.1%|-11.8% |
|BQSquare         |         -6.9% | -7.9% |  -8.1%|-10.2% |
|BlowingBubbles   |         -7.1% | -7.8% |  -8.1%|-10.7% |
|RaceHorses       |         -8.6% | -9.5% |  -9.6%|-13.1% |
|FourPeople       |         -9.9% | -11.1%| -11.9%|-14.9% |
|Johnny           |         -8.9% | -10.3%| -11.1%|-15.0% |
|KristenAndSara   |         -9.2% | -10.3%| -11.0%|-14.4% |
|Average          |         -7.3% | -8.2% | -8.6% |-10.9% |

**The run-time complexity:**

 The operating system is Windows Server 2012 R2 Standard and the memory is 256 GB. The table below shows the average run-time complexity in each class, and all of the time tests are conducted without GPU acceleration.
 
|Enc T[s] |HM 16.0 |Residual-VRN|
|:---:    |:---:   |:---:       |
|Class A  |1240.37 |23522.46    |
|Class B  |1027.21 |22040.25    |
|Class C  |539.06  |8998.16     |
|Class D  |220.56  |2241.11     |
|Class E  |631.57  |13316.99    | 

## Dataset Preparation
 Download HEVC standard video sequences and encode them.
 Arrange them in this way:
```
|YUV
|-- ClassA
|   |-- PeopleOnStreet_2560x1600_30.yuv
|   |-- Traffic_2560x1600_30.yuv
|-- ClassB
|   |-- ...
|-- ...
|pred
|-- ClassA
|   |-- qp22_AI
|   |   |-- PeopleOnStreet_2560x1600_30.yuv
|   |   |-- Traffic_2560x1600_30.yuv
|   |-- qp22_LD
|   |   |-- ...
|   |-- qp22_LDP
|   |   |-- ...
|   |-- qp22_RA
|   |   |-- ...
|   |-- qp27_AI
|   |   |-- ...
|   |-- ...
|   |-- qp37_RA
|   |   |-- ...
|rec
|-- ClassA
|   |-- qp22_AI
|   |-- | -- ...
|   |-- ...
|-- ...
|-- ClassB
|   |-- ...
```

## Using Residual-VRN
```bash
CUDA_VISIBLE_DEVICES='your deviced ids' python test.py --qp 22 --batch_size 4 --Class A
```
If you want to test on other datasets, you can use the function 'predict' in test.py. 
```angular2
def predict(pred_path, rec_path, predict_path, model_path, batch_size, n_frames):
    """
    :param pred_path: the path of predicted frames
    :param rec_path:  the path of reconstructed frames
    :param predict_path: the path of result
    :param model_path: model path
    :param batch_size: batch size
    :param n_frames: nums of the frames
    :return: None
    """
```




  
