class Solution:
    def searchInsert(self, nums, target):

        start = 0
        end = len(nums) - 1
        mid = 0

        while start <= end:

            mid = start + (end - start) // 2

            if nums[mid] == target:
                return mid

            elif nums[mid] < target:
                start = mid + 1

            else:
                end = mid - 1

        return mid + 1 if target > nums[mid] else mid

class Solution:
    def canReach(self, s: str, minJ: int, maxJ: int) -> bool:
        n = len(s)

        if int(s[-1]): return False

        dp = [False] * n
        dp[0] = True
        reach, maxR = 0, maxJ

        for i in range(minJ, n):
            if i > maxR: return False

            reach += dp[i - minJ]

            if i > maxJ:
                reach -= dp[i - maxJ - 1]

            if reach and not int(s[i]):
                dp[i] = True
                maxR = i + maxJ

        return reach > 0

class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lower = 0
        upper = 0

        for ch in word:
            if ch.islower():
                lower |= (1 << (ord(ch) - ord('a')))
            else:
                upper |= (1 << (ord(ch) - ord('A')))

        common = lower & upper

        # counting number of set bits
        return common.bit_count()

class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        A = [[False, False] for _ in range(27)]

        for ch in word:
            i = ord(ch) & 31
            c = ord(ch) >> 5 & 1
            A[i][c] = not (c and A[i][0])

        return sum(u and v for u, v in A)


class TrieNode:
    __slots__ = ['children', 'bestLen', 'bestIdx']
    
    def __init__(self):
        self.children = {}
        self.bestLen = float('inf')
        self.bestIdx = float('inf')

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        root = TrieNode()
        
        for i, word in enumerate(wordsContainer):
            n = len(word)
            curr = root
            
            if n < curr.bestLen or (n == curr.bestLen and i < curr.bestIdx):
                curr.bestLen = n
                curr.bestIdx = i
                
            for char in reversed(word):
                if char not in curr.children:
                    curr.children[char] = TrieNode()
                
                curr = curr.children[char]
                
                if n < curr.bestLen or (n == curr.bestLen and i < curr.bestIdx):
                    curr.bestLen = n
                    curr.bestIdx = i
                    
        ans = []
        
        for query in wordsQuery:
            curr = root
            
            for char in reversed(query):
                if char not in curr.children:
                    break
                curr = curr.children[char]
            
            ans.append(curr.bestIdx)
            
        return ans
class Solution:
    def minElement(self, nums: list[int]) -> int:
        min_val = float('inf')
        
        for num in nums:
            current_sum = 0
            
            while num > 0:
                current_sum += num % 10
                num //= 10
            
            min_val = min(min_val, current_sum)
                
        return min_val


class Solution:

    MAXX = 50000

    def __init__(self):
        self.seg = [0] * (4 * (self.MAXX + 1))

    def update(self, node, l, r, idx, val):
        if l == r:
            self.seg[node] = val
            return

        mid = (l + r) // 2

        if idx <= mid:
            self.update(2 * node, l, mid, idx, val)
        else:
            self.update(2 * node + 1, mid + 1, r, idx, val)

        self.seg[node] = max(
            self.seg[2 * node],
            self.seg[2 * node + 1]
        )

    def query(self, node, l, r, ql, qr):
        if ql > r or qr < l:
            return 0

        if ql <= l and r <= qr:
            return self.seg[node]

        mid = (l + r) // 2

        return max(
            self.query(2 * node, l, mid, ql, qr),
            self.query(2 * node + 1, mid + 1, r, ql, qr)
        )

    def getResults(self, queries: List[List[int]]) -> List[bool]:
        
        obstacles = SortedSet([0])

        # Build final obstacle configuration
        for q in queries:
            if q[0] == 1:
                obstacles.add(q[1])

        pos = list(obstacles)

        # gap[pos[i]] = pos[i] - pos[i-1]
        for i in range(1, len(pos)):
            self.update(1,0,self.MAXX,pos[i],pos[i] - pos[i - 1])

        ans = []

        for i in range(len(queries) - 1, -1, -1):

            if queries[i][0] == 2:

                x = queries[i][1]
                sz = queries[i][2]

                idx = obstacles.bisect_right(x) - 1
                prev_obstacle = obstacles[idx]

                best = self.query(1,0,self.MAXX,0,prev_obstacle)
                best = max(best, x - prev_obstacle)

                ans.append(best >= sz)

            else:

                x = queries[i][1]

                idx = obstacles.index(x)
                left_pos = obstacles[idx - 1]

                # remove gap ending at x
                self.update(1,0,self.MAXX,x,0)

                if idx + 1 < len(obstacles):
                    right_pos = obstacles[idx + 1]
                    # merge gaps
                    self.update(1,0,self.MAXX,right_pos,right_pos - left_pos)

                obstacles.remove(x)

        return ans[::-1]
        

class Solution:
    def asteroidsDestroyed(self, mass: int, asteroids: List[int]) -> bool:
        xmax=max(asteroids)
        freq=[0]*(1+xmax)
        for x in asteroids:
            freq[x]+=1
        planet=mass
        for x, f in enumerate(freq):
            if f==0:
                continue
            if x>planet:
                return False
            planet+=x*f
        return True


class Solution:
    def minimumCost(self, cost: List[int]) -> int:
        cost.sort()
        total = 0

        i = len(cost) - 1

        while i >= 0:
            total += cost[i]
            if i - 1 >= 0:
                total += cost[i - 1]

            i -= 3

        return total


class Solution:
    def calFinishTime(self, ls, ld, ws, wd):

        mini = float('inf')

        for i in range(len(ls)):
            mini = min(mini, ls[i] + ld[i])

        ans = float('inf')

        for i in range(len(ws)):
            ans = min(
                ans,
                max(mini, ws[i]) + wd[i]
            )

        return ans

    def earliestFinishTime(self, landStartTime: List[int], landDuration: List[int], waterStartTime: List[int], waterDuration: List[int]) -> int:
        return min(
            self.calFinishTime(landStartTime,landDuration,waterStartTime,waterDuration),
            self.calFinishTime(waterStartTime,waterDuration,landStartTime,landDuration)
        )


class Solution:
    def earliestFinishTime(
        self, la: list[int], lb: list[int], wa: list[int], wb: list[int]
    ) -> int:
        MAX = 300005
        l = w = minL = minW = MAX
        n, m = len(la), len(wa)

        for i in range(n):
            l = min(l, la[i] + lb[i])

        for i in range(m):
            w = min(w, wa[i] + wb[i])
            minL = min(minL, max(wa[i], l) + wb[i])

        for i in range(n):
            minW = min(minW, max(la[i], w) + lb[i])

        return min(minW, minL)

class Solution:
    MAX = 100001
    dp = [0] * MAX
    pref = [0] * MAX

    for i in range(100, MAX):
        r = i % 10
        m = (i // 10) % 10
        l = (i // 100) % 10

        isWave = m > max(l, r) or m < min(l, r)
        dp[i] = dp[i // 10] + int(isWave)
        pref[i] = pref[i - 1] + dp[i]

    def totalWaviness(self, A: int, B: int) -> int:
        return self.pref[B] - self.pref[A - 1]


class Solution:
    waves = []
    for i in range(1000):
        r = i % 10
        m = (i // 10) % 10
        l = (i // 100) % 10
        if (m > max(l, r)) | (m < min(l, r)):
            waves.append(i)

    def totalWaviness(self, A: int, B: int) -> int:
        return self.waveCount(B) - self.waveCount(A - 1)

    def waveCount(self, num):
        if num < 100: return 0
        return sum(self.countWays(num, p) for p in self.waves)

    def countWays(self, num, pattern):
        s = str(num)
        n = len(s)
        t = pattern < 100
        count = 0
        for i in range(n - 2):
            pre = int(s[:i] or 0)
            cur = int(s[i:i+3])
            suf = int(s[i+3:] or 0)
            mult = 10 ** (n - i - 3)
            ways = 0

            if cur > pattern:
                ways = pre - t + 1
            elif cur == pattern:
                ways = max(0, pre - t)
                count += suf + 1                
            else:
                ways = max(0, pre - t)
            count += ways * mult

        return count

class Solution:
    def leftRightDifference(self, nums: List[int]) -> List[int]:
        return (L:=list(accumulate(nums, initial=0))) and [abs(L[-1]-x-2*l) for l, x in zip(L, nums)]

class Solution:
    def createBinaryTree(self, A: List[List[int]]) -> Optional[TreeNode]:
        nodes = {}
        root = 0

        for x, y, is_left in A:
            if x not in nodes:
                nodes[x] = TreeNode(x)
                root ^= x
            if y not in nodes:
                nodes[y] = TreeNode(y)
                root ^= y
            if is_left:
                nodes[x].left = nodes[y]
            else:
                nodes[x].right = nodes[y]
            root ^= y

        return nodes[root]

class Solution:
    def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
        return (
            [n for n in nums if n < pivot] +
            [n for n in nums if n == pivot] +
            [n for n in nums if n > pivot]
        )
        

class Solution:
    def maxTotalValue(self, A: List[int], k: int) -> int:
        gMin = gMax = A[0]

        for n in A:
            gMin = min(gMin, n)
            gMax = max(gMax, n)

        return (gMax - gMin) * k