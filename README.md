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

|Type|Num. of Groups|Users|batch|embed|Sparsity|Recall@20|Precision@20|Hit@20|NDCG@20|best epoch|
|------|---|---|---|---|---|---|---|---|---|---|
|No clustering|1|29858|4096|64|0.00084|0.16183|0.04969|**0.55201**|**0.23384**|960|
|No clustering|1|29858|2048|64|0.00084|**0.16204**|**0.04988**|0.55174|0.23243|630|
|No clustering|1|29858|1024|64|0.00084|||
|Cluster 1|5|2474|4096|64|0.00396|0.24938|0.05610|0.62207|0.26898|1040|
|Cluster 1|5|2474|1024|64|0.00396|**0.27525**|**0.06180**|**0.66087**|**0.28455**|1020|
|Cluster 2|5|115|64|128|0.09082|**0.39303**|**0.06913**|**0.76522**|**0.23240**|160|
|Cluster 3|5|334|128|128|0.03037|**0.19472**|**0.05449**|**0.59581**|**0.19294**|440|
|Cluster 4|5|7188|4096|64|0.00269|0.15787|0.04553|0.53534|0.22357|810|
|Cluster 4|5|7188|2048|64|0.00269|0.16012|0.04612|0.54048|**0.22587**|680|
|Cluster 4|5|7188|1024|64|0.00269|**0.16022**|**0.04645**|**0.54243**|0.22505|520|
|Cluster 5|5|19747|4096|64|0.00117|**0.14286**|**0.04794**|**0.52975**|**0.22321**|850|
|Cluster 5|5|19747|2048|64|0.00117|0.14276|0.04803|0.52955|0.22241|710|
|Cluster 5|5|19747|1024|64|0.00117|0.14441|0.04873|0.52919|0.22192|590|
|Weighted Average|5|||||**0.15955**|**0.04888**|**0.54531**|**0.22863**||
|Cluster 1|4|21108|4096|64|0.00111|**0.15438**|**0.04938**|**0.54212**|**0.22780**|1090|
|Cluster 1|4|21108|1024|64|0.00111|0.15500|0.04980|0.51460|0.22676|600|
|Cluster 2|4|7982|4096|64|0.00249|**0.16841**|**0.04726**|**0.55162**|**0.23455**|1120|
|Cluster 2|4|7982|2048|64|0.00249|0.16702|0.04674|0.54924|0.23225|730|
|Cluster 2|4|7982|1024|64|0.00249|0.16736|0.04686|0.54823|0.23140|470|
|Cluster 3|4|333|1024|64|0.03060|0.16473|0.04775|0.52252|0.17523|980|
|Cluster 3|4|333|256|64|0.03060|0.16751|0.04715|0.54655|0.17109|360|
|Cluster 3|4|333|128|64|0.3060|0.19373|0.05450|0.60661|0.18762|620|
|Cluster 3|4|333|64|64|0.03060|**0.19500**|**0.05450**|**0.58595**|**0.19663**|410|
|Cluster 4|4|435|4096|64|0.01105|0.26875|0.05908|0.60460|0.22908|110|
|Cluster 4|4|435|1024|64|0.01105|**0.27393**|**0.06057**|**0.61839**|**0.24147**|310|
|Weighted Average|4|||||**0.16352**|**0.04903**|**0.54626**|**0.22946**


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
