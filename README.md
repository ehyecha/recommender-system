
### Overview
이 프로젝트는 Kaggle의 [The Movies Dataset](https://www.kaggle.com/code/elcior/starter-the-movies-dataset-fa37c888-3/input)을 활용하여 사용자의 영화 평점을 
예측하는 시스템을 개발하는 것입니다.
Collaborative Filtering 및 Content-based Filtering을 사용하며, Neural Matrix Factorization Model를 사용합니다.

### Dataset

- 포함된 주요 파일
* movies_metadata.csv: 영화 정보 (장르, 개봉 연도 등)
* ratings.csv: 사용자 평점 데이터
* keywords.csv: 영화의 키워드 정보

### Methodologies
Collaborative Filtering (User-based)
사용자 간 유사도를 기반으로 평점을 예측


Collaborative Filtering (Item-based)
영화 간 유사도를 기반으로 평점을 예측


Content-based Filtering
영화의 장르, 감독, 출연진 등의 정보를 활용하여 추천


Neural Matrix Factorization Model
신경망을 이용한 행렬 분해 모델을 활용
