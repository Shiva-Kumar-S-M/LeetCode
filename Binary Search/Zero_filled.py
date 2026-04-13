from collections import defaultdict
import math
from typing import List


class Solution:
    def countZeroFilledSubarrays(self,nums):
        count=0
        res=0
        for num in nums:
            if num==0:
                count+=1
                res+=count
            else:
                count=0
        return res


class Solution:
    def canPartitionGrid(self, grid: List[List[int]]) -> bool:
        m, n = len(grid), len(grid[0])
        total = sum(sum(row) for row in grid)
        
        if total % 2:
            return False
        
        target = total // 2
        s = 0
        
        for i in range(m - 1):
            s += sum(grid[i])
            if s == target:
                return True
        
        s = 0
        
        for j in range(n - 1):
            for i in range(m):
                s += grid[i][j]
            if s == target:
                return True
        
        return False


class Solution:
    def canPartitionGrid(self, grid):
        m, n = len(grid), len(grid[0])

        total = 0
        bottom = defaultdict(int)
        top = defaultdict(int) # type: ignore
        left = defaultdict(int) # type: ignore
        right = defaultdict(int)

        # Initialize bottom and right maps
        for row in grid:
            for x in row:
                total += x
                bottom[x] += 1
                right[x] += 1

        sumTop = 0

        # Horizontal cuts
        for i in range(m - 1):
            for j in range(n):
                val = grid[i][j]
                sumTop += val

                top[val] += 1
                bottom[val] -= 1

            sumBottom = total - sumTop

            if sumTop == sumBottom:
                return True

            diff = abs(sumTop - sumBottom)

            if sumTop > sumBottom:
                if self.check(top, grid, 0, i, 0, n - 1, diff):
                    return True
            else:
                if self.check(bottom, grid, i + 1, m - 1, 0, n - 1, diff):
                    return True

        sumLeft = 0

        # Vertical cuts
        for j in range(n - 1):
            for i in range(m):
                val = grid[i][j]
                sumLeft += val

                left[val] += 1
                right[val] -= 1

            sumRight = total - sumLeft

            if sumLeft == sumRight:
                return True

            diff = abs(sumLeft - sumRight)

            if sumLeft > sumRight:
                if self.check(left, grid, 0, m - 1, 0, j, diff):
                    return True
            else:
                if self.check(right, grid, 0, m - 1, j + 1, n - 1, diff):
                    return True

        return False

    def check(self, mp, grid, r1, r2, c1, c2, diff):
        rows = r2 - r1 + 1
        cols = c2 - c1 + 1

        # single cell
        if rows * cols == 1:
            return False

        # 1D row
        if rows == 1:
            return grid[r1][c1] == diff or grid[r1][c2] == diff

        # 1D column
        if cols == 1:
            return grid[r1][c1] == diff or grid[r2][c1] == diff

        return mp.get(diff, 0) > 0

class Solution:
    def areSimilar(self, mat, k):
        m, n = len(mat), len(mat[0])
        
        k %= n  # (reduce k<n)
        
        for i in range(m):
            for j in range(n):
                if i % 2 == 0:
                    # even row , left shift
                    if mat[i][j] != mat[i][(j + k) % n]:
                        return False
                else:
                    # odd row , right shift
                    if mat[i][j] != mat[i][(j - k) % n]:
                        return False
        
        return True

class Solution:
    def findTheString(self, lcp: List[List[int]]) -> str:
        n = len(lcp)
        word = [""] * n
        current = ord("a")

        for i in range(n):
            if not word[i]:
                if current > ord("z"):
                    return ""
                word[i] = chr(current)
                for j in range(i + 1, n):
                    if lcp[i][j]:
                        word[j] = word[i]
                current += 1

        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if word[i] != word[j]:
                    if lcp[i][j]:
                        return ""
                else:
                    if i == n - 1 or j == n - 1:
                        if lcp[i][j] != 1:
                            return ""
                    else:
                        if lcp[i][j] != lcp[i + 1][j + 1] + 1:
                            return ""

        return "".join(word)


class Solution:
    def checkStrings(self, s1: str, s2: str) -> bool:
        freq = [0] * 52

        for i, (a, b) in enumerate(zip(s1, s2)):
            off = (i & 1) * 26
            freq[ord(a) - 97 + off] += 1
            freq[ord(b) - 97 + off] -= 1

        return all(c == 0 for c in freq)

class Solution:
    def generateString(self, s: str, t: str) -> str:
        n, m = len(s), len(t)
        ans = ['?'] * (n + m - 1)  # ? indicates a pending position
        
        # Process 'T'
        for i, b in enumerate(s):
            if b != 'T':
                continue
            # The substring must match t
            for j, c in enumerate(t):
                v = ans[i + j]
                if v != '?' and v != c:
                    return ""
                ans[i + j] = c
        
        old_ans = ans
        ans = ['a' if c == '?' else c for c in ans]  # Initial default is 'a'
        
        # Process 'F'
        for i, b in enumerate(s):
            if b != 'F':
                continue
            # Substring must not equal t
            if ''.join(ans[i: i + m]) != t:
                continue
            # Locate the last pending position to modify
            for j in range(i + m - 1, i - 1, -1):
                if old_ans[j] == '?':  # Change 'a' to 'b'
                    ans[j] = 'b'
                    break
            else:
                return ""
        return ''.join(ans)

class Solution:
    def survivedRobotsHealths(self, positions, healths, directions):

        n = len(positions)
        order = sorted(range(n), key=lambda i: positions[i])

        h = healths[:]
        alive = [True]*n
        stack = []

        for idx in order:

            if directions[idx] == 'R':
                stack.append(idx)

            else:
                while stack:

                    top = stack[-1]

                    if h[top] < h[idx]:
                        alive[top] = False
                        stack.pop()
                        h[idx] -= 1

                    elif h[top] > h[idx]:
                        alive[idx] = False
                        h[top] -= 1
                        break

                    else:
                        alive[top] = False
                        alive[idx] = False
                        stack.pop()
                        break

        return [h[i] for i in range(n) if alive[i]]


class Solution:
    def decodeCiphertext(self, encodedText: str, rows: int) -> str:
        if rows == 1:
            return encodedText

        n = len(encodedText)
        cols = n // rows
        res = []

        for c in range(cols):
            r, j = 0, c
            while r < rows and j < cols:
                res.append(encodedText[r * cols + j])
                r += 1
                j += 1

        return "".join(res).rstrip()

class Solution:
    def judgeCircle(self, moves: str) -> bool:
        if len(moves) & 1: return False
        x, y = [0, 0]

        dir = {
            'U': (0, 1),
            'D': (0, -1),
            'L': (-1, 0),
            'R': (1, 0),
        }

        for c in moves:
            dx, dy = dir[c]
            x += dx
            y += dy

        return [x, y] == [0, 0]

class Solution:
    def robotSim(self, commands, obstacles):
        # Store obstacles
        blocked = set()
        for o in obstacles:
            blocked.add((o[0], o[1]))

        # Directions: North, East, South, West
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0)
        ]

        x, y = 0, 0
        dir = 0  # initially facing North
        maxDist = 0

        for cmd in commands:
            if cmd == -1:
                dir = (dir + 1) % 4  # turn right
            elif cmd == -2:
                dir = (dir + 3) % 4  # turn left
            else:
                while cmd > 0:
                    nx = x + directions[dir][0]
                    ny = y + directions[dir][1]

                    # check obstacle
                    if (nx, ny) in blocked:
                        break

                    x = nx
                    y = ny

                    maxDist = max(maxDist, x * x + y * y)
                    cmd -= 1

        return maxDist


# Added using AI
class Robot:
    def __init__(self, width: int, height: int):
        self.x = 0
        self.y = 0
        self.dir = "East"
        self.width = width
        self.height = height

    def step(self, num: int) -> None:
        perim = 2 * (self.width - 1) + 2 * (self.height - 1)
        num %= perim
        if num == 0:
            num = perim

        while num > 0:
            if self.dir == "East":
                maxX = min(self.x + num, self.width - 1)
                rem  = num - (maxX - self.x)
                num  = rem
                if rem == 0: self.x = maxX
                else:        self.x = maxX; self.dir = "North"
            elif self.dir == "West":
                minX = max(self.x - num, 0)
                rem  = num - (self.x - minX)
                num  = rem
                if rem == 0: self.x = minX
                else:        self.x = minX; self.dir = "South"
            elif self.dir == "North":
                maxY = min(self.y + num, self.height - 1)
                rem  = num - (maxY - self.y)
                num  = rem
                if rem == 0: self.y = maxY
                else:        self.y = maxY; self.dir = "West"
            elif self.dir == "South":
                minY = max(self.y - num, 0)
                rem  = num - (self.y - minY)
                num  = rem
                if rem == 0: self.y = minY
                else:        self.y = minY; self.dir = "East"

    def getPos(self): return [self.x, self.y]
    def getDir(self): return self.dir


    class Solution:
        def xorAfterQueries(self, nums, queries):
            mod = 1000000007

        # Process each query
            for t in queries:
                l = t[0]
                r = t[1]
                k = t[2]
                v = t[3]

                idx = l

            # Apply operation at step k
                while idx <= r:
                    temp = nums[idx]
                    nums[idx] = (temp * v) % mod
                    idx += k

        # Compute XOR of final array
            ans = 0
            for num in nums:
                ans ^= num

            return ans

class Solution:
    def xorAfterQueries(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        limit = math.isqrt(n)
        
        # Group queries with small k for later processing
        lightK = defaultdict(list)
        
        for q in queries:
            l, r, k, v = q
            
            if k >= limit:
                # Large k: apply brute force
                for i in range(l, r + 1, k):
                    nums[i] = (nums[i] * v) % MOD
            else:
                # Small k: process later
                lightK[k].append(q)
                
        for k, query_list in lightK.items():
            # Process small queries grouped by step size k
            diff = [1] * n
            
            for q in query_list:
                l, r, _, v = q
                
                # Multiply starting position
                diff[l] = (diff[l] * v) % MOD
                
                # Cancel the multiplication using modular inverse
                steps = (r - l) // k
                nxt = l + (steps + 1) * k
                if nxt < n:
                    # pow(v, -1, MOD) computes the modular inverse natively
                    diff[nxt] = (diff[nxt] * pow(v, -1, MOD)) % MOD
                    
            # Propagate the multipliers with a step size of k
            for i in range(n):
                if i >= k:
                    diff[i] = (diff[i] * diff[i - k]) % MOD
                nums[i] = (nums[i] * diff[i]) % MOD
                
        ans = 0
        for num in nums:
            ans ^= num
            
        return ans


class Solution:
    def minimumDistance(self, nums: List[int]) -> int:
        n = len(nums)
        last2 = [0] * n
        res = 200

        for i in range(n):
            val, pos = nums[i] - 1, i + 1
            pack = last2[val]
            old, cur = pack & 255, pack >> 8

            last2[val] = cur | (pos << 8)

            if old:
                res = min(res, (pos - old) << 1)

        return -(res == 200) | res

class Solution:
    def minimumDistance(self, nums: List[int]) -> int:
        n, M=len(nums), max(nums)
        pos=[(-1, -1) for _ in range(M+1)]
        ans=1<<32
        for k, x in enumerate(nums):
            if pos[x][1]!=-1:
                ans=min(ans, (k-pos[x][1])<<1)
            pos[x]=k, pos[x][0]
        return -1 if ans==1<<32 else ans  

# dp[i][j]=min cost after typing word[i], with the OTHER finger at j
# 26 denotes hovering
class Solution:
    def minimumDistance(self, word: str) -> int:
        def dist(x, y):
            if x==26 or y==26: return 0
            return abs(x//6-y//6)+abs(x%6-y%6)
        # setting for dp
        n=len(word)
        INF=1<<30
        dp=[[INF]*27 for _ in range(n)]
        dp[0][26]=0
        prev=ord(word[0])-65
        for i, c in enumerate(word[1:], start=1):
            x=ord(c)-65
            for j in range(27):
                dp[i][j]=min(dp[i][j], dp[i-1][j]+dist(prev, x))
                dp[i][prev]=min(dp[i][prev], dp[i-1][j]+dist(j, x))
            prev=x
        return min(dp[-1])
        
class Solution:
    def getMinDistance(self, nums: list[int], target: int, start: int) -> int:

        if nums[start] == target:
            return 0

        n = len(nums)
        d = 1

        while True:
            if start - d >= 0 and nums[start - d] == target:
                return d

            if start + d < n and nums[start + d] == target:
                return d

            d += 1    