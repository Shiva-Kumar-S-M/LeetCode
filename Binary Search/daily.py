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

class Solution:
    def countSubmatrices(self, grid: List[List[int]], k: int) -> int:
        r, c=len(grid), len(grid[0])
        cnt, brCol=0, c
        if grid[0][0]>k:
            return 0
        cnt+=1
        for j in range(1, c):
            grid[0][j]+=grid[0][j-1]
            if grid[0][j]>k:
                brCol=j
                break
            cnt+=1
        for i in range(1, r):
            grid[i][0]+=grid[i-1][0]
            if grid[i][0]>k:
                break
            cnt+=1
            for j in range(1, brCol):
                grid[i][j]+=grid[i-1][j]+grid[i][j-1]-grid[i-1][j-1]
                if grid[i][j]>k:
                    brCol=j
                    break
                cnt+=1
        return cnt