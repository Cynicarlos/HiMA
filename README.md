# HiMA ([arXiv](https://arxiv.org/abs/2510.15497))

Clone this repository
```
git clone https://github.com/Cynicarlos/HiMA.git
cd HiMA
```
## Environment Preparation

### Use packed environment
To make it easy, we provide the packed environemnt, you can just download `HiMA.tar.gz` from [Google Drive](https://drive.google.com/file/d/1Jx7gSDNOdY4Mkddc_bokeaGfqLZ7FYTl/view?usp=sharing) or [Pan Baidu](https://pan.baidu.com/s/1DD3DC0m2-dYNe54C3dBknA?pwd=tbjd), and upload it to the environment folder (eg. `~/anaconda3/envs` or `~/miniconda/envs`) and unzip it as follows.
```
cd ~/anaconda3/envs
mkdir HiMA
mv HiMA.tar.gz HiMA
cd HiMA
tar -zxvf HiMA.tar.gz
```
Then activate it by `conda activate HiMA`.

### Create environment from scratch
```
conda create -n HiMA python=3.9
conda activate HiMA
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```
Then `pip install -r requirements.txt`

If errors occur when install mamba_ssm, please download the wheel file in the following link and than pip install it offline.

[mamba-ssm_baidu](https://pan.baidu.com/s/1K-FeQBXwf6hnGqa6xmqYdw?pwd=f8df) / [mamba-ssm_google](https://drive.google.com/file/d/1LAepdNvy4iCpQCpv7mR5hK1Q8gecx0l6/view?usp=sharing) or [mamba-ssm_other_versions](https://github.com/state-spaces/mamba/releases)



If you want to use Muon optimizer, install it as follows, else ignore it.

`
pip install git+https://github.com/KellerJordan/Muon
`

## Dataset Preparation
| Dataset | Download link |  Source  |  CFA     |
| :---:   |    :----:     |  :---:   |  :---:   |
| Sony    | [Google Drive](https://drive.google.com/file/d/1G6VruemZtpOyHjOC5N8Ww3ftVXOydSXx/view)       | [Link](https://github.com/cchen156/Learning-to-See-in-the-Dark)   |  Bayer  |
| Fuji    | [Google Drive](https://drive.google.com/file/d/1C7GeZ3Y23k1B8reRL79SqnZbRBc4uizH/view)       | [Link](https://github.com/cchen156/Learning-to-See-in-the-Dark)   |  X-Trans  |
| MCR     | [Google Drive](https://drive.google.com/file/d/1Q3NYGyByNnEKt_mREzD2qw9L2TuxCV_r/view)       | [Link](https://github.com/TCL-AILab/Abandon_Bayer-Filter_See_in_the_Dark)   |  Bayer  |
| ELD     | [Google Drive](https://drive.google.com/drive/folders/1QoEhB1P-hNzAc4cRb7RdzyEKktexPVgy)       | [Link](https://github.com/Vandermode/ELD)   |  Bayer  |

The directory for the datasets should be as following:  

```
📁datasets/
├─── 📁ELD/
│    └─── 📁SonyA7S2/
│         ├─── 📄scene1_0001.ARW
│         └─── 📄...
│    └─── 📁NikonD850/
│         ├─── 📄scene1_0001.nef
│         └─── 📄...
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
torchrun --nproc_per_node=1 train.py --config=configs/sony.yaml --use_muon
```
Otherwise, use 
```
python train.py --config=configs/sony.yaml
```
The results should be similar but the training time is longer for Muon.

For other datasets, just change the ```config```. Please always remember to change the ```data_dir``` of ```xxx.yaml```  to the right place for both training and testing.

If you want to run the program in the background, you can use the script ```./train.sh``` Remember to install ```tmux``` first. And when you firstly try this, you should usually run ```chmod +x train.sh```

Then you may use ```tail -f train.log``` to see the training process.
## Test
Before evaluating our pretrained models, please download them by the following links and change their name to `sony.pth` et. al and put them in the ```pretrained``` folder.  
[Google Drive](https://drive.google.com/drive/folders/196hPm0aLqpgsxLryKqKpE0UKgvXE_0ap?usp=drive_link) or [Baidu Drive](https://pan.baidu.com/s/146zs6nfFdNcTmA3ytsd7vQ?pwd=8fem).

```
python test_sony.py
```

## Citation
If there is any help for your research, please star this repository and if you want to follow this work, you can cite as follows:
```md
@misc{chen2025rethinkingefficienthierarchicalmixing,
      title={Rethinking Efficient Hierarchical Mixing Architecture for Low-light RAW Image Enhancement}, 
      author={Xianmin Chen and Peiliang Huang and Longfei Han and Dingwen Zhang and Junwei Han},
      year={2025},
      eprint={2510.15497},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.15497}, 
}
```
