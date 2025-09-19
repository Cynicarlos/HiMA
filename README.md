# HiMA

## Environment Preparation
```
pip install -r requirements.txt
pip install git+https://github.com/KellerJordan/Muon
```

## Dataset Preparation
| Dataset | Download link |  Source  |  CFA     |
| :---:   |    :----:     |  :---:   |  :---:   |
| Sony    | [Google Drive](https://drive.google.com/file/d/1G6VruemZtpOyHjOC5N8Ww3ftVXOydSXx/view)       | [Link](https://github.com/cchen156/Learning-to-See-in-the-Dark)   |  Bayer  |
| Fuji    | [Google Drive](https://drive.google.com/file/d/1C7GeZ3Y23k1B8reRL79SqnZbRBc4uizH/view)       | [Link](https://github.com/cchen156/Learning-to-See-in-the-Dark)   |  X-Trans  |
| MCR     | [Google Drive](https://drive.google.com/file/d/1Q3NYGyByNnEKt_mREzD2qw9L2TuxCV_r/view)       | [Link](https://github.com/TCL-AILab/Abandon_Bayer-Filter_See_in_the_Dark)   |  Bayer  |

The directory for the datasets should be as following:  

```
📁datasets/  
├─── 📁MCR/  
│    ├─── 📄MCR_test_list.txt  
│    ├─── 📄MCR_train_list.txt  
│    └─── 📁Mono_Colored_RAW_Paired_DATASET/  
│         ├─── 📁Color_RAW_Input/
│         │    ├─── 📄C00001_48mp_0x8_0x00ff.tif
│         │    └─── 📄...
│         └─── 📁RGB_GT/
│              ├─── 📄C00001_48mp_0x8_0x2fff.jpg
│              └─── 📄...
└─── 📁SID/  
     ├─── 📁Fuji/  
     │    ├─── 📄Fuji_test_list.txt  
     │    ├─── 📄Fuji_train_list.txt  
     │    ├─── 📄Fuji_val_list.txt  
     │    └─── 📁Fuji/  
     │         ├─── 📁Long/
     │         │    ├─── 📄00001_00_10s.RAF
     │         │    └─── 📄...
     │         └─── 📁Short/
     │              ├─── 📄00001_00_0.1s.RAF
     │              └─── 📄...
     └─── 📁Sony/  
          ├─── 📄Sony_test_list.txt  
          ├─── 📄Sony_train_list.txt  
          ├─── 📄Sony_val_list.txt  
          └─── 📁Sony/  
               ├─── 📁Long/
               │    ├─── 📄00001_00_10s.ARW
               │    └─── 📄...
               └─── 📁Short/
                    ├─── 📄00001_00_0.1s.ARW
                    └─── 📄...
```

## Train
If you want to use ```Muon``` optimizer, please use ```torchrun``` as follows:
```
torchrun --nproc_per_node=1 train.py --config=configs/sony.yaml
```
Otherwise, use 
```
python train.py --config=configs/sony.yaml
```
The results should be similar.

For other datasets, just change the ```config```.

If you want to run the program in the background, you can use the script ```./train.sh``` Remember to install ```tmux``` first. And when you firstly try this, you should usually run ```chmod +x train.sh```

Then you may use ```tail -f train.log``` to see the training process.
## Test
Before evaluating our pretrained models, please download them by the following links and put them in the ```pretrained``` folder.  
[Google Drive](https://drive.google.com/drive/folders/196hPm0aLqpgsxLryKqKpE0UKgvXE_0ap?usp=drive_link) or [Baidu Drive](https://pan.baidu.com/s/146zs6nfFdNcTmA3ytsd7vQ?pwd=8fem)
```
python test_sony.py
```

## Citation
If there is any help for your research, please star this repository and if you want to follow this work, you can cite as follows:
```md
xxx
```
