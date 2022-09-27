# Chunk-Level Attention SER
This is a Keras implementation of chunk-level speech emotion recognition (SER) framework in the [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9442335) for the MSP-Podcast corpus.

:exclamation::exclamation::exclamation:**NEW: If you are looking for PyTorch implementation, please go to this [repo](https://github.com/winston-lin-wei-cheng/Chunk-Level-Attention-SER-PyTorch)**

:exclamation::exclamation::exclamation:**NEW: If you are looking for applying dynamic-chunk-segmentation as general data preprocessing step, please go to this [repo](https://github.com/winston-lin-wei-cheng/Dynamic-Chunk-Segmentation)**

![The Chunk-Level Attention SER Framework](/images/framework.png)

# Suggested Environment and Requirements
1. Python 3.6
2. Ubuntu 18.04
3. keras version 2.2.4
4. tensorflow version 1.14.0
5. CUDA 10.0
6. The scipy, numpy and pandas...etc conventional packages
7. The MSP-Podcast corpus (request to download from [UTD-MSP lab website](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html))
8. The IS13ComParE LLDs (acoustic features) extracted by OpenSmile (users can refer to the [opensmile-LLDs-extraction](https://github.com/winston-lin-wei-cheng/opensmile-LLDs-extraction) repository) 

# How to run
After extracted the IS13ComParE LLDs (e.g., XXX_llds/feat_mat/\*.mat) for MSP-Podcast *[whatever version]* corpus, we use the *'labels_concensus.csv'* provided by the corpus as the default input label setting. 

1. change data & label root paths in **norm_para.py**, then run it to get z-norm parameters (mean and std) based on the Train set. We also provide the parameters of the v1.6 corpus in the *'NormTerm'* folder.

2. change data & label root paths in **lstm_training.py** or **cnn1D_training.py** for LSTM or CNN-based model, the running args are,
   * -ep: number of epochs
   * -batch: batch size for training
   * -emo: emotion attributes (Act, Dom or Val)
   * -atten: type of chunk-level attneiton model (NonAtten, GatedVec, RnnAttenVec or SelfAttenVec)
   * run in the terminal
```
python lstm_training.py -ep 100 -batch 128 -emo Act -atten RnnAttenVec
```

3. change data & label & model root paths in **lstm_testing.py** or **cnn1D_testing.py** for the testing results based on the MSP-Podcast test set,
   * run in the terminal
```
python lstm_testing.py -ep 100 -batch 128 -emo Act -atten RnnAttenVec
```

# Pre-trained models
We provide some trained model weights based on **version 1.6** of the MSP-Podcast in the *'trained_model_v1.6'* folder. The CCC performances of models are the same as the [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9442335).

| Model            | Act              | Val              | Dom              | Online [ms/uttr] |
|:----------------:|:----------------:|:----------------:|:----------------:|:----------------:|
| LSTM-RnnAttenVec | 0.6955           | 0.3006           | 0.6175           | 40.9             |
| LSTM-SelfAttenVec| 0.6837           | 0.3337           | 0.6004           | 41.3             |
| CNN-NonAtten     | 0.7035           | 0.2683           | 0.6268           | 1.8              |
| CNN-GatedVec     | 0.7027           | 0.2856           | 0.6201           | 3.0              |

Users can get these results by running the lstm_testing.py and cnn1D_testing.py with corresponding args.


# For general usage
The implementation is for the MSP-Podcast corpus, however, the framework can be applied on general speech-based sequence-to-one tasks (e.g., speaker recognition, gender detection, acoustic event classification or SER...etc). If you want to apply the framework on your own tasks, here are some important parameters need to be specified in the **DynamicChunkSplitTrainingData/DynamicChunkSplitTestingData** functions under the **utils.py** file,
1. max duration in second of your corpus (i.e., Tmax)
2. desired chunk window length in second (i.e., Wc)
3. number of chunks splitted in a sentence (i.e., C = ceiling of Tmax/Wc)
4. number of frames within a chunk (i.e., m)
5. scaling factor to increase the splited chunks number (i.e., n=1, 2 or 3 are suggested)
6. remember to change NN model dimensions: feat_num, time_step and C


# Reference
If you use this code, please cite the following paper:

Wei-Cheng Lin and Carlos Busso, "Chunk-Level Speech Emotion Recognition: A General Framework of Sequence-to-One Dynamic Temporal Modeling"

```
@article{Lin_202x,
    author={W.-C. Lin and C. Busso},
    title={Chunk-Level Speech Emotion Recognition: A General Framework of Sequence-to-One Dynamic Temporal Modeling},
    journal={IEEE Transactions on Affective Computing},
    number={},
    volume={To Appear},
    pages={},
    year={2021},
    month={},
    doi={10.1109/TAFFC.2021.3083821},
}
```
