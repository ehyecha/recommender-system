
from MatrixFactorization import MatrixFactorization 
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np

class CollaborativeFiltering:
  def __init__(self, user_movie_matrix, k= 10):
    self.user_movie_matrix = user_movie_matrix
    self.dense_matrix = None
    self.optimal_k = k

    self.user_similarity = None
    self.item_similarity = None
    self.user_means = None


  def calculate_pearson_similarity(self, user_id):
      if self.dense_matrix is None:
        mf = MatrixFactorization(self.user_movie_matrix)
        reduced_matrix = mf.fit(self.optimal_k)
        self.dense_matrix = csr_matrix(reduced_matrix)
      
      reduced_matrix_csr = self.dense_matrix
      target_user_vector = reduced_matrix_csr[user_id].toarray().flatten() 
      similarities = {}

      for other_user_id in range(reduced_matrix_csr.shape[0]):
        if other_user_id != user_id:
            other_user_vector = reduced_matrix_csr[other_user_id].toarray().flatten()  

            # 공통으로 평가한 아이템만 선택 (0 제외)
            common_indices = (target_user_vector != 0) & (other_user_vector != 0)
            if np.sum(common_indices) > 1:  # 공통 아이템이 2개 이상일 때만 계산
                corr, _ = pearsonr(target_user_vector[common_indices], other_user_vector[common_indices])
                similarities[other_user_id] = corr
            else:
                similarities[other_user_id] = 0  # 공통 평가 데이터가 없으면 0 처리

      self.user_similarity = similarities
      return similarities

  def adjusted_cosine_similarity(self):
      """
      아이템 간의 Adjusted Cosine Similarity를 계산하여 item_similarity 행렬을 저장
        """
      if self.dense_matrix is None:
        mf = MatrixFactorization(self.user_movie_matrix)
        reduced_matrix = mf.fit(self.optimal_k)
        self.dense_matrix = csr_matrix(reduced_matrix)
      user_means = np.array([np.nanmean(row[row > 0]) if np.any(row > 0) else 0 for row in self.dense_matrix.toarray()])


      # ✅ 조정된 행렬 생성
      adjusted_matrix = self.dense_matrix.copy()
      adjusted_matrix.data = adjusted_matrix.data - user_means[adjusted_matrix.indices]

      # ✅ Adjusted Cosine Similarity 계산
      item_similarity = cosine_similarity(adjusted_matrix, dense_output=False)

      # ✅ 저장 및 반환
      self.item_similarity = item_similarity
      return self.item_similarity

  def user_based_predict(self, user_id, item_name, similarities, k=10):
    """
    특정 사용자와 아이템에 대해 평점을 예측.

    Parameters:
        user_id (int): 예측할 사용자 ID
        item_name (int): 예측할 아이템 
        similarities (dict): 사용자 간 유사도 딕셔너리
        k (int): 유사한 사용자 수
    Returns:
        float: 예측 평점
    """
    item_id = self.user_movie_matrix.columns.get_loc(item_name)
    rated_users = [user for user in similarities if not pd.isna(self.user_movie_matrix.iloc[user, item_id])]

    # 유사도 기준으로 해당 아이템에 대해 평가한 사용자 중 상위 k명 선택
    top_k_users = sorted([(user, similarities[user]) for user in rated_users], key=lambda x: x[1], reverse=True)[:k]
    # 사용자 평균 평점 계산
    user_mean = self.user_movie_matrix.loc[user_id]
    #user_mean = user_mean[user_mean != 0].mean()  # 0이 아닌 값만 평균 계산
    user_mean = user_mean[user_mean != 0].mean() if not user_mean[user_mean != 0].empty else 0
    numerator = 0
    denominator = 0
    for similar_user, sim in top_k_users:
        if item_id in self.user_movie_matrix.columns:
            # 유사 사용자의 평균 평점
            similar_user_mean = self.user_movie_matrix.loc[similar_user].mean()

            # 유사 사용자의 평점과 평균 차이
            rating_diff = self.user_movie_matrix.loc[similar_user, item_id] - similar_user_mean
            numerator += sim * rating_diff
            denominator += abs(sim)
    # 평점 예측
    if denominator == 0:
        return user_mean  # 유사한 사용자가 없을 경우 평균 평점 반환
    return user_mean + (numerator / denominator)

  def item_based_predict(self,item_index, user_id, k= 10):
    """
      특정 사용자와 아이템에 대해 평점을 예측

      Parameters:
        item_index (str): 아이템 인덱스
        user_id (int): 사용자 ID
        top_n (int): 유사한 아이템 수

      Returns:
        float: 예측 평점
      """
    sim_vector = self.item_similarity.getrow(item_index).toarray().flatten()

    # 자기 자신은 추천에서 제외: 유사도를 매우 낮게 만들어 무시되도록 함
    sim_vector[item_index] = 0
    # 유사도가 높은 순서대로 후보 아이템 인덱스 정렬
    candidate_indices = np.argsort(sim_vector)[::-1]
    recommendations = []
    cluster_matrix = self.user_movie_matrix
    for idx in candidate_indices:
        # ratings_matrix에서 해당 아이템(idx)의 평점 데이터가 있는지 확인 (nnz > 0이면 평점 존재)
      if self.dense_matrix.getrow(idx).nnz > 0 and cluster_matrix.iloc[idx].loc[user_id]:
        recommendations.append((idx, cluster_matrix.iloc[idx].loc[user_id], sim_vector[idx]))
        # 충분한 추천 아이템을 찾으면 종료
      if len(recommendations) >= k:
        break
    index = cluster_matrix.columns.get_loc(user_id)
    user_ratings = cluster_matrix.iloc[:,index]
    user_mean = user_ratings[user_ratings != 0].mean()
    numerator = 0
    denominator = 0
    for item, rating, sim in recommendations:
          numerator += sim * rating
          denominator += abs(sim)
    if denominator == 0:
        return user_mean  # 유사한 사용자가 없을 경우 평균 평점 반환
    return numerator / denominator if denominator != 0 else 0