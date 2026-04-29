from bisect import bisect_left
from collections import defaultdict
import math
import string
from typing import List

from numpy import inf
from sympy import product


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



class Solution:
    def minimumTotalDistance(self, robot: List[int], factory: List[List[int]]) -> int:
        robot.sort()
        factory.sort()

        n, m = len(robot), len(factory)
        INF = float('inf')

        dp = [[INF]*(m+1) for _ in range(n+1)]

        for j in range(m+1):
            dp[0][j] = 0

        for j in range(1, m+1):
            pos, limit = factory[j-1]

            for i in range(n+1):
                dp[i][j] = dp[i][j-1]

                dist = 0
                for k in range(1, min(limit, i)+1):
                    dist += abs(robot[i-k] - pos)
                    dp[i][j] = min(dp[i][j], dp[i-k][j-1] + dist)

        return dp[n][m]


class Solution:
    def maxDistance(self, A: List[int], B: List[int]) -> int:
        i, j = 0, 1

        while i < len(A) and j < len(B):
            i += A[i] > B[j]
            j += 1

        return j - i - 1
    

class Solution:
    def maxDistance(self, A: List[int]) -> int:
        n = len(A)
        left, right = 0, n - 1

        for i in range(n):
            if A[i] ^ A[-1]:
                left = i
                break

        for i in range(n - 1, -1, -1):
            if A[i] ^ A[0]:
                right = i
                break

        return max(n - 1 - left, right)

class Solution:
    def minimumHammingDistance(self, source: List[int], target: List[int], allowedSwaps: List[List[int]]) -> int:
        n = len(source)

        parent = list(range(n))
        rank = [0] * n

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def unite(a, b):
            pa, pb = find(a), find(b)
            if pa == pb:
                return

            if rank[pa] < rank[pb]:
                pa, pb = pb, pa

            parent[pb] = pa
            if rank[pa] == rank[pb]:
                rank[pa] += 1

        for a, b in allowedSwaps:
            unite(a, b)

        from collections import defaultdict

        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        ans = 0

        for idxs in groups.values():
            freq = {}

            for i in idxs:
                freq[source[i]] = freq.get(source[i], 0) + 1

            for i in idxs:
                if freq.get(target[i], 0) > 0:
                    freq[target[i]] -= 1
                else:
                    ans += 1

        return ans


class Solution:
    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)
		
    def find(self, a):
        if self.parent[a] != a:
            self.parent[a] = self.find(self.parent[a])

        return self.parent[a]
        
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
		# 1. Union-Find
        self.parent = list(range(len(s)))
        for a, b in pairs:
            self.union(a, b)

		# 2. Grouping
        group = defaultdict(lambda: ([], []))  
        for i, ch in enumerate(s):
            parent = self.find(i)
            group[parent][0].append(i)
            group[parent][1].append(ch)

		# 3. Sorting
        res = [''] * len(s)
        for ids, chars in group.values():
            ids.sort()
            chars.sort()
            for ch, i in zip(chars, ids):
                res[i] = ch
                
        return ''.join(res)


class Solution:
    def twoEditWords(self, queries: List[str], dictionary: List[str]) -> List[str]:
        def _get_distance(s1,s2):
            cnt = 0
            for i in range(len(s1)):
                if s1[i]!=s2[i]:
                    cnt+=1
                if cnt==3:
                    return False
            return True

        good = []
        for query in queries:
            for d in dictionary:
                dist = _get_distance(query,d)
                if dist:
                    good.append(query)
                    break
        return good
    
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordSet = set(wordList)
        if endWord not in wordSet:
            return 0
        
        begin_set = {beginWord}
        end_set = {endWord}
        visited = set()
        steps = 1
        
        while begin_set and end_set:
            if len(begin_set) > len(end_set):
                begin_set, end_set = end_set, begin_set
            
            next_set = set()
            
            for word in begin_set:
                for i in range(len(word)):
                    for c in string.ascii_lowercase:
                        if c == word[i]:
                            continue
                        
                        new_word = word[:i] + c + word[i+1:]
                        
                        if new_word in end_set:
                            return steps + 1
                        
                        if new_word in wordSet and new_word not in visited:
                            visited.add(new_word)
                            next_set.add(new_word)
            
            begin_set = next_set
            steps += 1
        
        return 0

class Solution:
    def distance(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = [0] * n

        mp = defaultdict(list)

        for i, v in enumerate(nums):
            mp[v].append(i)

        for pos in mp.values():
            total = sum(pos)
            left_sum = 0
            m = len(pos)

            for i in range(m):
                right_sum = total - left_sum - pos[i]

                left = pos[i] * i - left_sum
                right = right_sum - pos[i] * (m - i - 1)

                ans[pos[i]] = left + right

                left_sum += pos[i]

        return ans


class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        x, r=0, 0
        for c in moves:
            x+=(c=='R')-(c=='L')
            r+=c=='_'
        return abs(x)+r


class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        res = []
        for x, y in points:
            if x == 0:
                res.append(y)
            elif y == side:
                res.append(side + x)
            elif x == side:
                res.append(side * 3 - y)
            else:
                res.append(side * 4 - x)
        res.sort()
        def check(n : int) -> bool:
            idx = [0] * k
            curr = res[0]
            for i in range(1, k):
                j = bisect_left(res, curr + n)
                if j == len(res):
                    return False
                idx[i] = j
                curr = res[j]
            if curr - res[0] <= side * 4 - n:
                return True
            
            for idx[0] in range(1, idx[1]):
                for j in range(1, k):
                    while res[idx[j]] < res[idx[j - 1]] + n:
                        idx[j] += 1
                        if idx[j] == len(res):
                            return False
                if res[idx[-1]] - res[idx[0]] <= side * 4 - n:
                    return True
            return False
        
        left, right = 1, side + 1
        while left + 1 < right:
            mid = (left + right) // 2
            if check(mid):
                left = mid
            else:
                right = mid
        return left
        

class Solution:        
    def maxPointsInsideSquare(self, points: List[List[int]], s: str) -> int:
        minLens = {}
        secondMin = float('inf')
        
        for point, char in zip(points, s):
            size = max(abs(point[0]), abs(point[1]))

            if char not in minLens:
                minLens[char] = size
            elif size < minLens[char]:
                secondMin = min(minLens[char], secondMin)
                minLens[char] = size
            else:
                secondMin = min(size, secondMin)
        
        count = 0
        for len in minLens.values():
            if len < secondMin:
                count += 1
        
        return count

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        m, n = len(grid), len(grid[0])
        visit = [False] * (m * n)
        dirs = ((0, -1), (0, 1), (-1, 0), (1, 0))

        def dfs(r, c, pr, pc):
            visit[r * n + c] = True

            for dr, dc in dirs:
                nr, nc = r + dr, c + dc

                if (nr, nc) != (pr, pc):
                    if 0 <= nr < m and 0 <= nc < n:
                        if grid[nr][nc] == grid[r][c]:
                            if visit[nr * n + nc] or dfs(nr, nc, r, c):
                                return True

            return False

        return any(not visit[i] and dfs(i // n, i % n, -1, -1) for i in range(m * n))


class Solution:
    TRANS = [
        [-1, 1, -1, 3],
        [0, -1, 2, -1],
        [3, 2, -1, -1],
        [1, -1, -1, 2],
        [-1, 0, 3, -1],
        [-1, -1, 1, 0]
    ]
    DIRS = [[-1, 0], [0, 1], [1, 0], [0, -1]]
    START = [[1, 3], [0, 2], [2, 3], [1, 2], [0, 3], [0, 1]]

    def hasValidPath(self, grid: List[List[int]]) -> bool:       
        m, n = len(grid), len(grid[0])
        if m == 1 and n == 1: return True

        def check(d):
            if d == -1: return False
            r, c = self.DIRS[d]
            # O(1) Space
            while 0 <= r < m and 0 <= c < n:               
                d = self.TRANS[grid[r][c] - 1][d]
                if d == -1: return False
                if r == 0 and c == 0: return False
                if r == m - 1 and c == n - 1: return True
                
                dr, dc = self.DIRS[d] 
                r += dr
                c += dc
            return False

        a, b = self.START[grid[0][0] - 1]
        return check(a) or check(b)


class Solution:
    def minOperations(self, grid: List[List[int]], x: int) -> int:
        n, m = len(grid), len(grid[0])
        N = n * m
        freq = [0] * 10001
        mn = grid[0][0]
        mx = mn

        for row in grid:
            for c in row:
                if (c - grid[0][0]) % x != 0: return -1
                freq[c] += 1
                mn = min(mn, c)
                mx = max(mx, c)

        target = (N + 1) // 2
        acc = 0
        median = mn

        for i in range(mn, mx + 1, x):
            acc += freq[i]
            if acc >= target:
                median = i
                break

        ops = 0
        for i in range(mn, mx + 1, x):
            ops += abs(i - median) // x * freq[i]

        return ops

class Solution:
    def minMoves2(self, nums: List[int]) -> int:
        nums.sort()
        median = nums[len(nums) // 2]

        moves = 0
        for num in nums:
            moves += abs(num - median)

        return moves


class Solution:
    def maximumScore(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])
        if m == 1:
            return 0

        # prefix sum per column
        col = [[0]*(n+1) for _ in range(m)]
        for j in range(m):
            for i in range(n):
                col[j][i+1] = col[j][i] + grid[i][j]

        dp = [[0]*(n+1) for _ in range(n+1)]
        prefMax = [[0]*(n+1) for _ in range(n+1)]
        suffMax = [[0]*(n+1) for _ in range(n+1)]

        for c in range(1, m):

            newdp = [[0]*(n+1) for _ in range(n+1)]

            for curr in range(n+1):
                for prev in range(n+1):

                    if curr <= prev:
                        gain = col[c][prev] - col[c][curr]

                        newdp[curr][prev] = max(
                            newdp[curr][prev],
                            suffMax[prev][0] + gain
                        )
                    else:
                        gain = col[c-1][curr] - col[c-1][prev]

                        newdp[curr][prev] = max(
                            newdp[curr][prev],
                            suffMax[prev][curr],
                            prefMax[prev][curr] + gain
                        )

            # build prefix & suffix
            for curr in range(n+1):

                prefMax[curr][0] = newdp[curr][0]

                for prev in range(1, n+1):
                    penalty = 0
                    if prev > curr:
                        penalty = col[c][prev] - col[c][curr]

                    prefMax[curr][prev] = max(
                        prefMax[curr][prev-1],
                        newdp[curr][prev] - penalty
                    )

                suffMax[curr][n] = newdp[curr][n]

                for prev in range(n-1, -1, -1):
                    suffMax[curr][prev] = max(
                        suffMax[curr][prev+1],
                        newdp[curr][prev]
                    )

            dp = newdp

        ans = 0
        for k in range(n+1):
            ans = max(ans, dp[0][k], dp[n][k])

        return ans

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:

        ans, m, n = -inf, len(grid), len(grid[0])
        grid = [[math.inf]*n] + grid
        grid = [[inf]+row for row in grid]
        # for row in grid: print(row)
        
        for row, col in product(range(m), range(n)):
            mn = min(grid[row][col+1], grid[row+1][col])
            ans = max(ans, grid[row+1][col+1] - mn)
            grid[row+1][col+1] = min(grid[row+1][col+1], mn)
         # for row in grid: print(row)
        
        return ans