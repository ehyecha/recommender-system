import numpy as np
from sklearn.decomposition import TruncatedSVD

class MatrixFactorization:
  def __init__(self, user_matrix, max_k=500):
    self.user_matrix = user_matrix
    self.max_k = max_k
    self.best_k = None
    self.svd_model = None

  def fit(self, k):
    svd = TruncatedSVD(n_components=k, n_iter=5, random_state=42)
    self.svd_model = svd
    reduced_matrix = svd.fit_transform(self.user_matrix)
    return reduced_matrix

  def predict(self, k, top_k, is_user =True):
      reduced_matrix = self.fit(k)
      reconstructed_matrix = self.svd_model.inverse_transform(reduced_matrix)
      if is_user:
        actual_items = {user_id: np.where(self.user_matrix.iloc[user_id] > 0)[0] for user_id in range(self.user_matrix.shape[0])}
        predicted_items = {user_id: np.argsort(reconstructed_matrix[user_id])[::-1][:top_k]
                   for user_id in range(self.user_matrix.shape[0])}
      else:
        actual_items = {user_id: np.where(self.user_matrix.iloc[user_id] > 0)[0] for user_id in range(self.user_matrix.shape[1])}
        predicted_items = {item_id: np.argsort(reconstructed_matrix[item_id])[::-1][:top_k]
                   for item_id in range(self.user_matrix.shape[1])}
      return actual_items, predicted_items

  def precision_at_k(self, actual_items, predicted_items, k =10):
    precisions = []
    for user_id in actual_items.keys():
        actual_set = set(actual_items[user_id])
        predicted_set = set(predicted_items[user_id][:k])
        if len(actual_set) > 0:  # 실제 아이템이 없는 경우 제외
            precisions.append(len(actual_set & predicted_set) / k)
    return np.mean(precisions) if precisions else 0

  def recall_at_k(self, actual_items, predicted_items, k=10):
    recall_scores = []
    for user in actual_items:
        if len(actual_items[user]) == 0:
            continue
        pred_set = set(predicted_items[user][:k])
        actual_set = set(actual_items[user])
        recall = len(pred_set & actual_set) / len(actual_set)
        recall_scores.append(recall)
    return np.mean(recall_scores)