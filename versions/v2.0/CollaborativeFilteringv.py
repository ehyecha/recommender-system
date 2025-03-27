
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans
from MatrixFactorization import MatrixFactorization 

class CollaborativeFiltering:
  def __init__(self, user_movie_matrix, n_clusters, k, k_neighbors = 10):
      self.user_movie_matrix = user_movie_matrix
      self.optimal_k = k
      self.n_clusters = n_clusters
      self.k_neighbors = k_neighbors
      self.dense_matrix = None
      self.k_menas = None
      self.user_labels = None
      self.user_similarity = None
      self.item_similarity = None
      self.user_means = None
      self.nearest_users = None

  def clustering(self):
      """차원축소후 KMeans로 사용자 군집화"""
      user_matrix_sparse = csr_matrix(self.user_movie_matrix)
      svd = TruncatedSVD(n_components=self.optimal_k, random_state=42)
      X_reduced = svd.fit_transform(user_matrix_sparse)
      batch_size = min(1024, X_reduced.shape[0])  # 배치 크기 최적화
      self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=batch_size, n_init="auto")
      user_clusters = self.kmeans.fit_predict(X_reduced)
      self.user_labels = user_clusters

  def get_nearest_neighbors(self, user_index):
      """동일 클러스터 내에서 NearestNeighbors를 사용해 가장 가까운 유저 찾기"""
      user_cluster = self.user_labels[user_index]  # 현재 사용자가 속한 클러스터 찾기
      cluster_indices = [i for i, label in enumerate(self.user_labels) if label == user_cluster]

      # 클러스터 내 최근접 이웃 찾기
      knn = NearestNeighbors(n_neighbors=self.k_neighbors, metric='cosine')
      knn.fit(self.user_movie_matrix.iloc[cluster_indices].values)

      user_vector = self.user_movie_matrix.iloc[user_index].values.reshape(1, -1)
      distances, indices = knn.kneighbors(user_vector)

      nearest_users = [cluster_indices[i] for i in indices[0]]  # 원래 인덱스로 변환
      return nearest_users

  def calculate_pearson_similarity(self, user_id):

      if self.dense_matrix is None:
        mf = MatrixFactorization(self.user_movie_matrix)
        reduced_matrix = mf.fit(self.optimal_k)
        self.dense_matrix = csr_matrix(reduced_matrix)
      reduced_matrix_csr = self.dense_matrix
      
      user_index = self.user_movie_matrix.index.get_loc(user_id)
      target_user_vector = reduced_matrix_csr[user_index].toarray().flatten()
      nearest_users = self.get_nearest_neighbors(user_index)
      similarities = {}

      for other_user_id in nearest_users:
        if other_user_id != user_index:
            other_user_vector = reduced_matrix_csr[other_user_id].toarray().flatten()  
            # 공통으로 평가한 아이템만 선택 (0 제외)
            common_indices = (target_user_vector != 0) & (other_user_vector != 0)
            if np.sum(common_indices) > 1:  # 공통 아이템이 2개 이상일 때만 계산
                target_values = target_user_vector[common_indices]
                other_values = other_user_vector[common_indices]
                corr, _ = pearsonr(target_values, other_values)
                similarities[other_user_id] = corr
            else:
                similarities[other_user_id] = 0  # 공통 평가 데이터가 없으면 0 처리
      self.user_similarity = similarities
      return similarities

  def adjusted_cosine_similarity(self, item_index):
      """
      아이템 간의 Adjusted Cosine Similarity를 계산하여 item_similarity 행렬을 저장
        """
      if self.dense_matrix is None:
        mf = MatrixFactorization(self.user_movie_matrix)
        reduced_matrix = mf.fit(self.optimal_k)
        self.dense_matrix = csr_matrix(reduced_matrix)
      user_means = np.array([np.nanmean(row[row > 0]) if np.any(row > 0) else 0 for row in self.dense_matrix.toarray()])

      # ✅ 조정된 행렬 생성

      nearest_users = self.get_nearest_neighbors(item_index)
      self.nearest_users = nearest_users
      if len(nearest_users) == 0:
            print(f"경고: 아이템 {item_index}을 평가한 유저가 없습니다. 기본값 반환")
            return np.zeros(self.dense_matrix.shape[0])
      adjusted_matrix = self.dense_matrix[nearest_users,:].copy()
      user_means = np.array([np.nanmean(row[row > 0]) if np.any(row > 0) else 0 for row in self.dense_matrix.toarray()])

    # ✅ 조정된 행렬 생성
      adjusted_matrix.data = adjusted_matrix.data - user_means[adjusted_matrix.indices]

      # ✅ Adjusted ㄷCosine Similarity 계산
      item_similarity = cosine_similarity(adjusted_matrix, dense_output=False)

      # 조정된 코사인 유사도 계산
      self.item_similarity = item_similarity
      return self.item_similarity

  def user_based_predict(self, user_id, item_name, similarities):
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
    user_mean = self.user_movie_matrix.loc[user_id]
    self.user_mean = user_mean[user_mean != 0].mean() if not user_mean[user_mean != 0].empty else 0
    rated_users = [user for (user, value) in similarities.items() if self.user_movie_matrix.iloc[user, item_id] != 0
                  and value > 0]
    if len(rated_users) == 0:
      return self.user_mean
    # 유사도 기준으로 해당 아이템에 대해 평가한 사용자 유사도가 0이상인 사용자만을 선택
    top_k_users = sorted([(int(self.user_movie_matrix.index[user]), similarities[user]) for user in rated_users if similarities[user] > 0], key=lambda x: x[1], reverse=True)

    # 사용자 평균 평점 계산
    numerator = 0
    denominator = 0
    for similar_user, sim in top_k_users:
        rating = self.user_movie_matrix.loc[similar_user, item_name]
        if rating !=0:
            # 유사 사용자의 평균 평점
            similar_user_mean = self.user_movie_matrix.loc[similar_user]
            similar_user_mean = similar_user_mean[similar_user_mean != 0].mean() if not similar_user_mean[similar_user_mean != 0].empty else 0
            # 유사 사용자의 평점과 평균 차이
            rating_diff = self.user_movie_matrix.loc[similar_user, item_name] - similar_user_mean
            numerator += sim * rating_diff
            denominator += abs(sim)
    if denominator == 0:
        return self.user_mean  # 유사한 사용자가 없을 경우 평균 평점 반환
    return self.user_mean + (numerator / denominator)

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
    sim_vector = self.item_similarity[0].toarray()[0][1:]

    # # 유사도가 높은 순서대로 후보 아이템 인덱스 정렬
    candidate_indices = np.argsort(sim_vector)[::-1]
    recommendations = []
    cluster_matrix = self.user_movie_matrix
    for idx in candidate_indices:
        # ratings_matrix에서 해당 아이템(idx)의 평점 데이터가 있는지 확인 (nnz > 0이면 평점 존재)
      item_idx = self.nearest_users[idx]
      if self.dense_matrix.getrow(item_idx).nnz > 0 and cluster_matrix.iloc[item_idx].loc[user_id]:
        recommendations.append((idx, cluster_matrix.iloc[item_idx].loc[user_id], sim_vector[idx]))
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