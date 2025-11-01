# GRAPH-BASED EMOTION CONSENSUS PERCEPTION LEARNING FOR MULTIMODAL EMOTION RECOGNITION IN CONVERSATION

### Requirements

- Python 3.8.5
- torch 1.7.1
- CUDA 11.3
- torch-geometric 1.7.2

### Dataset

The raw data can be found at [IEMOCAP](https://sail.usc.edu/iemocap/ "IEMOCAP") and [MELD](https://github.com/SenticNet/MELD "MELD").

In our paper, we use pre-extracted features. The multimodal features (including RoBERTa-based and GloVe-based textual features) are available at [here](https://www.dropbox.com/sh/4b21lympehwdg4l/AADXMURD5uCECN_pvvJpCAy9a?dl=0 "here").

### Testing and Checkpoints

The implementation results may vary with training machines and random seeds. We suggest that one can try different random seeds for better results.

We also provide some pre-trained checkpoints on RoBERTa-based IEMOCAP at [here](https://www.dropbox.com/sh/gd32s36v7l3c3u9/AACOipUURd7gEbEcdYSrmP-0a?dl=0 "here").

For instance, to test on IEMOCAP using the checkponts:

`python -u train.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size 16 --graph_type='hyper' --epochs=0 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='IEMOCAP' --norm BN --testing`

### Training examples

To train on IEMOCAP:

`python -u train.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size 16 --graph_type='hyper' --epochs=80 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='IEMOCAP' --norm BN --num_L=3 --num_K=4`

To train on MELD:

`python -u train.py --base-model 'GRU' --dropout 0.4 --lr 0.0001 --batch-size 16 --graph_type='hyper' --epochs=15 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='MELD' --norm BN --num_L=3 --num_K=3`
