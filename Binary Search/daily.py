from ast import List


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

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        sumX = [0] * cols
        sumY = [0] * cols
        res = 0

        for i in range(rows):
            rx = 0
            ry = 0
            for j in range(cols):
                if grid[i][j] == 'X':
                    rx += 1
                elif grid[i][j] == 'Y':
                    ry += 1
                
                sumX[j] += rx
                sumY[j] += ry
                
                if sumX[j] > 0 and sumX[j] == sumY[j]:
                    res += 1

        return res

# Added using AI
class Solution:
    def minAbsDiff(self, grid: list[list[int]], k: int) -> list[list[int]]:
        m, n = len(grid), len(grid[0])
        ans = [[0] * (n - k + 1) for _ in range(m - k + 1)]

        for i in range(m - k + 1):
            for j in range(n - k + 1):
                v = sorted(set(
                    grid[x][y]
                    for x in range(i, i + k)
                    for y in range(j, j + k)
                ))
                if len(v) <= 1:
                    ans[i][j] = 0
                else:
                    ans[i][j] = min(v[p+1] - v[p] for p in range(len(v) - 1))

        return ans


class Solution:
    def reverseSubmatrix(self, grid: List[List[int]], x: int, y: int, k: int) -> List[List[int]]:
        for i in range(k):
            for j in range(k // 2):
                grid[x + j][y + i], grid[x + k - j - 1][y + i] = grid[x + k - j - 1][y + i], grid[x + j][y + i]
        return grid

        
class Solution:
    def findRotation(self, mat: List[List[int]], target: List[List[int]]) -> bool:
        for x in range(4):
            if mat == target: return True
            mat = [list(r) for r in zip(*mat[::-1])]
        return False

class Solution:
    def maxProductPath(self, grid: List[List[int]]) -> int:
        r, c, MOD=len(grid), len(grid[0]), 10**9+7
        dp=[[[0]*2 for _ in range(c)] for _ in range(r)]

        p=dp[0][0][0]=dp[0][0][1]=grid[0][0]
        for j in range(1, c):
            p*=grid[0][j]
            dp[0][j][0]=dp[0][j][1]=p

        p=grid[0][0]
        for i in range(1, r):
            p*=grid[i][0]
            dp[i][0][0]=dp[i][0][1]=p
            for j in range(1, c):
                x=grid[i][j]
                xx=(x*dp[i][j-1][0], x*dp[i][j-1][1], x*dp[i-1][j][0], x*dp[i-1][j][1])
                dp[i][j][0], dp[i][j][1]=min(xx), max(xx)
        ans=dp[r-1][c-1][1]
        return -1 if ans<0 else ans%MOD