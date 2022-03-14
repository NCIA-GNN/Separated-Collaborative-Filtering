# Separated-Collaborative-Filtering

## Pytorch-NGCF
아래 링크 참고하여 수정함    
Original Pytorch  Implementation can be found [here](https://github.com/liu-jc/PyTorch_NGCF)

## 진행상황
- NGCF 원본 동작 확인
- gowalla 기준 incidence_matrix -> spectral coclustering -> 1개 cluster에 대해 학습 진행
- ```utility/load_data.py``` 에서 ```def get_adj_mat```에 clustering / index rearrange 코드 작성(clustering 핵심코드)
- 1개의 cluster를 골라서 re-index해서 학습 진행
- 동시학습은 아직임
## gowalla dataset 중간결과(early stopping)

|Type|Users|batch|embed|Sparsity|NDCG@20|best epoch|
|------|---|---|---|---|---|---|
|No clustering|29858|4096|64|0.00084|**0.23384**|960|
|No clustering|29858|2048|64|0.00084|0.23243|630|
|Cluster 1|2474|4096|64|0.00396|0.26898|1040|
|Cluster 1|2474|1024|64|0.00396|**0.28455**|1020|
|Cluster 2|115|64|128|0.09082|**0.23240**|160|
|Cluster 3|334|128|128|0.03037|**0.19294**|440|
|Cluster 4|7188|4096|64|0.00269|0.22357|810|
|Cluster 4|7188|2048|64|0.00269|**0.22587**|680|
|Cluster 4|7188|1024|64|0.00269|0.22505|520|
|Cluster 5|19747|4096|64|0.00117|**0.22321**|850|
|Cluster 5|19747|2048|64|0.00117|0.22241|710|
|Cluster 5|19747|1024|64|0.00117|0.22192|590|
|Weighted Average|||||0.22863||

## Run the Code
### NGCF with Gowalla

**원본 대비 옵션 4개 추가** 
- ```--scc``` : Spectral Co-Clustering 적용 여부. 0 : 적용 / 1 : 미적용(원본)
- ```--create``` : 최초 ```*.npz```파일 생성 여부. 0 : 기존 npz load / 1 : npz파일 생성    
- ```--N``` : 몇 개의 cluster로 나눌 건지?
- ```--cl_num``` 몇 번째 cluster로 실험할건지

```
python main.py --dataset gowalla --alg_type ngcf --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 4096 --epoch 5000 --verbose 1 --mess_dropout [0.1,0.1,0.1] --create 0 --scc 0 --N 5 --cl_num 0

```
