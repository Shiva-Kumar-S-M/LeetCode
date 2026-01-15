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

class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix or not matrix[0]:
            return 0

        M = len(matrix)
        N = len(matrix[0])

        # Convert characters to integers
        for i in range(M):
            for j in range(N):
                matrix[i][j] = int(matrix[i][j])

        # Row-wise prefix widths
        for i in range(M):
            for j in range(1, N):
                if matrix[i][j] == 1:
                    matrix[i][j] += matrix[i][j - 1]

        Ans = 0

        # Fix each column
        for j in range(N):
            for i in range(M):
                width = matrix[i][j]
                if width == 0:
                    continue

                # Expand downward
                currWidth = width
                k = i
                while k < M and matrix[k][j] > 0:
                    currWidth = min(currWidth, matrix[k][j])
                    height = k - i + 1
                    Ans = max(Ans, currWidth * height)
                    k += 1

                # Expand upward
                currWidth = width
                k = i
                while k >= 0 and matrix[k][j] > 0:
                    currWidth = min(currWidth, matrix[k][j])
                    height = i - k + 1
                    Ans = max(Ans, currWidth * height)
                    k -= 1

        return Ans

class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        low, high, total_area = float('inf'), float('-inf'), 0

        for x, y, l in squares:
            total_area += l*l
            low = min(low, y)
            high = max(high, y+l)
        
        target_area = total_area / 2.0

        for i in range(60):
            mid = (low+high) / 2.0

            curr_area = 0
            for _, y, l in squares:
                curr_y = max(0, min(l, mid-y))
                curr_area += l*curr_y
            
            if curr_area < target_area:
                low = mid
            else:
                high = mid

        return mid

class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        events = []
        for x, y, l in squares:
            events.append((y, 1, x, x + l))
            events.append((y + l, -1, x, x + l))

        events.sort()
        xs = []
        prev_y = events[0][0]
        total = 0
        areas = []

        def union_len(intervals):
            intervals.sort()
            res = cur = 0
            end = -10**30
            for a, b in intervals:
                if a > end:
                    res += b - a
                    end = b
                elif b > end:
                    res += b - end
                    end = b
            return res

        for y, typ, x1, x2 in events:
            if y > prev_y and xs:
                h = y - prev_y
                w = union_len(xs)
                areas.append((prev_y, h, w))
                total += h * w
            if typ == 1:
                xs.append((x1, x2))
            else:
                xs.remove((x1, x2))
            prev_y = y

        half = total / 2
        acc = 0
        for y, h, w in areas:
            if acc + h * w >= half:
                return y + (half - acc) / w
            acc += h * w

        return 0.0

class Solution:
    def maximizeSquareHoleArea(self, n: int, m: int, hBars: list[int], vBars: list[int]) -> int:
        def maxSpan(bars: list[int]) -> int:
            bars.sort()
            res = 1
            streak = 1
            for i in range(1, len(bars)):
                if bars[i] - bars[i - 1] == 1:
                    streak += 1
                else:
                    streak = 1
                res = max(res, streak)
            return res + 1
        
        return min(maxSpan(hBars), maxSpan(vBars)) ** 2