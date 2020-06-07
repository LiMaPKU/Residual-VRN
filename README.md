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

## Test
```bash
CUDA_VISIBLE_DEVICES='your deviced ids' python test.py --qp 22 --batch_size 4 --Class A
```





  
