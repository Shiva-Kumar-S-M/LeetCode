class Solution:
    def getBiggestThree(self, grid):
        m, n = len(grid), len(grid[0])
        s = set()

        for i in range(m):
            for j in range(n):
                s.add(grid[i][j])

                k = 1
                while True:
                    if i-k<0 or i+k>=m or j-k<0 or j+k>=n:
                        break

                    total = 0

                    r, c = i-k, j
                    for t in range(k):
                        total += grid[r+t][c+t]

                    r, c = i, j+k
                    for t in range(k):
                        total += grid[r+t][c-t]

                    r, c = i+k, j
                    for t in range(k):
                        total += grid[r-t][c-t]

                    r, c = i, j-k
                    for t in range(k):
                        total += grid[r-t][c+t]

                    s.add(total)
                    k += 1

        return sorted(s, reverse=True)[:3]
    
class Solution:
    def largestSubmatrix(self, matrix: List[List[int]]) -> int:
        m = len(matrix)
        n = len(matrix[0])
        res = 0

        for i in range(1, m):
            for j in range(n):
                if matrix[i][j] == 1:
                    matrix[i][j] += matrix[i - 1][j]

        for i in range(m):
            matrix[i].sort(reverse=True)
            for j in range(n):
                res = max(res, matrix[i][j] * (j + 1))

        return res