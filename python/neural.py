import tenserflow as tf
from tenseflow.keras import layers,dense
model = Seqential([
    layers.Dense(64,activation='relu',input_shape=(32,)),
    layers.Dense(10,activation='softmax')
])

from collections import deque

class Solution:
    def subtreeWithAllDeepest(self, root):
        if not root:
            return None

        parent = {root: None}
        q = deque([root])

        last_level = []

        # BFS traversal
        while q:
            size = len(q)
            last_level = []
            for _ in range(size):
                node = q.popleft()
                last_level.append(node)

                if node.left:
                    parent[node.left] = node
                    q.append(node.left)
                if node.right:
                    parent[node.right] = node
                    q.append(node.right)

        # last_level contains all deepest nodes
        deepest = set(last_level)

        # Move up until they meet
        while len(deepest) > 1:
            deepest = {parent[node] for node in deepest}

        return deepest.pop()

class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        n, m = len(s1), len(s2)

        # dp[i][j] = maximum ASCII sum of common subsequence
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(n):
            for j in range(m):
                if s1[i] == s2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + ord(s1[i])
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

        total_ascii = sum(ord(c) for c in s1) + sum(ord(c) for c in s2)
        return total_ascii - 2 * dp[n][m]