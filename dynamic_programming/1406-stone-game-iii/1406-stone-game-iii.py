class Solution:
    s = ["Bob", "Tie", "Alice"]
    def stoneGameIII(self, A: list[int]) -> str:
        n = len(A)
        dp = [0, 0, 0, 0]

        for i in range(n - 1, -1, -1):
            dp[i & 3] = -5e7
            tot = 0

            for j in range(1, 4):
                if i + j <= n:
                    tot += A[i + j - 1]
                    dp[i & 3] = max(dp[i & 3], tot - dp[(i + j) & 3])

        return self.s[(dp[0] > 0) - (dp[0] < 0) + 1]