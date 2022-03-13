# Separated-Collaborative-Filtering

## Pytorch-NGCF
아래 링크 참고하여 수정함    
Original Pytorch  Implementation can be found [here](https://github.com/liu-jc/PyTorch_NGCF)

### 진행상황
- NGCF 원본 동작 확인
- gowalla 기준 incidence_matrix -> spectral coclustering -> 1개 cluster에 대해 학습 진행
- test도중 에러..
- ```utility/load_data.py``` 에서 ```def get_adj_mat```에 clustering / index rearrange 코드 작성

### Run the Code
#### NGCF with Gowalla

**원본 대비 옵션 2개 추가** 
- ```--scc``` : Spectral Co-Clustering 적용 여부. 0 : 적용 / 1 : 미적용(원본)
- ```--create``` : 최초 ```*.npz```파일 생성 여부. 0 : 기존 npz load / 1 : npz파일 생성    

```
python main.py --dataset gowalla --alg_type ngcf --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 4096 --epoch 500 --verbose 1 --mess_dropout [0.1,0.1,0.1] --scc 0 --create 0

```
