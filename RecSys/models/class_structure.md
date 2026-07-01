# Переиспользуемые функции и описания ошибок

```python
def _check_is_fitted(self) -> None:
    if not self.is_fitted:
        raise RuntimeError(
            "The model must be fitted via '.fit()' before prediction"
        )
```

```python
def _validate_user(self, user: np.ndarray) -> None:
    if user.ndim != 1:
        raise ValueError("'user' must be a one-dimensional array")
    if user.shape[0] != self.n_items:
        raise ValueError(
            f"'user' must contain {self.n_items} items; got {user.shape[0]}"
        )
```

```python
def _validate_users(self, users: np.ndarray) -> None:
    if users.ndim != 2:
        raise ValueError(f"'users' must be two-dimensional; got shape {users.shape}")
    if users.shape[1] != self.n_items:
        raise ValueError(
            f"'users' must contain {self.n_items} items; got {users.shape[1]}"
        )
```

```python
@staticmethod
def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if not 1 <= k <= scores.size:
        raise ValueError(f"'k' must be between 1 and {scores.size}; got {k} instead")

    topk_indices = np.argpartition(scores, -k)[-k:]
    topk_indices = topk_indices[np.argsort(scores[topk_indices], descending=True)]

    return topk_indices
```

```python
@staticmethod
def _time_decay_weights(timedelta: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * timedelta)
```


```python
self._check_is_fitted()

user = np.asarray(user)
self._validate_user(user)

seen_items = user > 0
user = seen_items.astype(np.float64)

scores = self.similarity_scores(user)
scores[seen_items] = -np.inf

indices = self._topk_indices(scores=scores, k=prediction_size)

if return_scores:
    return indices, scores[indices]

return indices
```


