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


class Solution:
    def maxTotalValue(self, nums: list[int], k: int) -> int:
        n = len(nums)
        LUT = SparseTable(nums)

        pq = [(-LUT.query(i, n), i, n) for i in range(n)]

        res = 0
        for _ in range(k):
            val, l, r = pq[0]
            if val == 0:
                break
            res -= val
            heapq.heapreplace(pq, (-LUT.query(l, r - 1), l, r - 1))

        return res

class SparseTable:
    def __init__(self, num: list[int]):
        n = len(num)
        bitWidth = n.bit_length()
        self.Min = [[0] * n for _ in range(bitWidth)]
        self.Max = [[0] * n for _ in range(bitWidth)]

        for i in range(n):
            self.Min[0][i] = self.Max[0][i] = num[i]

        for i in range(1, bitWidth):
            for j in range(n - (1 << i) + 1):
                self.Min[i][j] = min(self.Min[i - 1][j], self.Min[i - 1][j + (1 << (i - 1))])
                self.Max[i][j] = max(self.Max[i - 1][j], self.Max[i - 1][j + (1 << (i - 1))])

    def query(self, left: int, right: int) -> int:
        k = (right - left).bit_length() - 1
        return max(self.Max[k][left], self.Max[k][right - (1 << k)]) - \
               min(self.Min[k][left], self.Min[k][right - (1 << k)])
    

class Solution:
    def assignEdgeWeights(self, edges: List[List[int]]) -> int:
        mod = 1_000_000_007
        n = len(edges) + 1
        graph = [[] for _ in range(n + 1)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        def dfs(node: int, prev: int) -> int:
            d = 0
            for c in graph[node]:
                if c != prev:
                    d = max(d, dfs(c, node) + 1)
            return d

        return pow(2, dfs(1, 0) - 1, mod)


import numpy as np
import collections
import sys

class Solution:
    def assignEdgeWeights(self, edges: List[List[int]], queries: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        MOD = 10**9 + 7
        LOG_N = 18 

        # 1. Ultra-fast flat list adjacency structure
        adj = [[] for _ in range(n + 1)]
        for u, v in edges:
            adj[u].append(v); adj[v].append(u)

        up = [[0] * LOG_N for _ in range(n + 2)]
        depth = [0] * (n + 2)

        # 2. Iterative BFS Queue (Replaces slow recursion)
        # Element format: (current_node, parent_node, current_depth)
        queue = collections.deque([(1, 0, 0)])
        visited = [False] * (n + 1)
        visited[1] = True

        while queue:
            node, parent, d = queue.popleft()
            depth[node] = d
            up[node][0] = parent
            
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, node, d + 1))

        # 3. Compute Binary Lifting Matrix 
        for j in range(1, LOG_N):
            for i in range(1, n + 1):
                prev = up[i][j-1]
                up[i][j] = up[prev][j-1] if prev != 0 else 0

        # =========================================================================
        # VECTORIZED VECTOR MATRIX PROCESSING
        # =========================================================================
        depth_np = np.array(depth, dtype=np.int32)
        up_np = np.array(up, dtype=np.int32)
        queries_np = np.array(queries, dtype=np.int32)

        u, v = queries_np[:, 0], queries_np[:, 1]
        orig_u, orig_v = u.copy(), v.copy()

        # Step A: Leveling
        swap_mask = depth_np[u] < depth_np[v]
        u[swap_mask], v[swap_mask] = v[swap_mask], u[swap_mask]

        diff = depth_np[u] - depth_np[v]
        for j in range(LOG_N):
            jump_mask = (diff >> j) & 1 == 1
            u[jump_mask] = up_np[u[jump_mask], j]

        # Step B: Climbing
        for j in range(LOG_N - 1, -1, -1):
            jump_mask = (u != v) & (up_np[u, j] != up_np[v, j])
            u[jump_mask] = up_np[u[jump_mask], j]
            v[jump_mask] = up_np[v[jump_mask], j]

        lca = u.copy()
        not_equal_mask = (u != v)
        lca[not_equal_mask] = up_np[u[not_equal_mask], 0]

        # Step C: Combinatorial Parity math
        path_lengths = depth_np[orig_u] + depth_np[orig_v] - 2 * depth_np[lca]
        
        # Optimized lookup approach: precompute powers array using NumPy vectorization
        pow2 = np.zeros(n + 2, dtype=np.int64)
        pow2[0] = 1
        for i in range(1, n + 2):
            pow2[i] = (pow2[i-1] * 2) % MOD

        # Direct NumPy index extraction (Avoids slower element-by-element lambda maps)
        ans = np.where(path_lengths == 0, 0, pow2[(path_lengths - 1).astype(np.int32)])

        return ans.tolist()

class Solution:
    def mapWordWeights(self, words: List[str], wt: List[int]) -> str:
        res = []

        for word in words:
            s = 0
            for ch in word:
                s += wt[(ord(ch) & (1 << 5) - 1) - 1]
            res.append(chr(122 - (s - ((s * 2521) >> (1 << 4)) * len(wt))))

        return "".join(res)

class Solution:
    def pairSum(self, head: Optional[ListNode]) -> int:
        slow = fast = head
        prev = None

        while fast and fast.next:
            fast = fast.next.next
            slow.next, prev, slow = prev, slow, slow.next

        res = 0
        while slow:
            res = max(res, prev.val + slow.val)
            prev, slow = prev.next, slow.next

        return res

class Solution:
    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head.next: return None

        slow = head
        fast = slow.next.next

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        slow.next = slow.next.next
        return head

class Solution:
    """
        Processes a string based on specific character commands:
        - Lowercase letters: Added to the result.
        - '*': Removes the last element (Backspace functionality).
        - '#': Duplicates the current result set.
        - '%': Reverses the current result set.
        
        Complexity Analysis:
        - Time: O(N * K), where N is the number of characters and K 
          is the length of the list during '%' or '#' operations.
        - Space: O(N) to store the result list.
        """
    def processStr(self, s: str) -> str:
        result = []

        for c in s:
            if c.islower():
                result.append(c)
            elif c == '*':
                if result:
                    result.pop()
            elif c == '#':
                # Production note, this was the fastest way to do the doubling of the string
                result += result 

                # Production note: Using the built-in .reverse() method
                # is the most efficient 'best practice' way to perform
                # an in-place reversal in Python.
            elif c == '%':
                result.reverse()
        
        return "".join(result)

class Solution:
    def processStr(self, s: str, k: int) -> str:
        n = len(s)
        lens = []
        ln = 0

        for c in s:
            if c == '*':
                ln = max(ln - 1, 0)
            elif c == '#':
                ln *= 2
            elif c != '%':
                ln += 1
            
            lens.append(ln)

        if k >= ln:
            return '.'

        for i in range(n - 1, -1, -1):
            c = s[i]
            if c == '*':
                continue
            elif c == '#':
                if k >= lens[i] // 2:
                    k -= lens[i] // 2
            elif c == '%':
                k = lens[i] - 1 - k
            else:
                if lens[i] == k + 1:
                    return c


class Solution:
    def angleClock(self, hour: int, minutes: int) -> float:
        x = hour + minutes / 60
        diff = (11 * x) % 12
        return min(diff, 12 - diff) * 30


class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        ans = acc = 0

        for it in gain:
            acc += it
            d = acc - ans
            ans += d & ~(d >> 0x1F)

        return ans


class Solution:
    def maxBuilding(self, num: int, r: list[list[int]]) -> int:
        r.append([1, 0])
        r.sort()
        n = len(r)

        def yCap(x1, y1, x2, y2):
            return min(y2, y1 + abs(x2 - x1))

        def yPeak(x1, y1, x2, y2):
            return (y1 + y2 + x2 - x1) >> 1
        
        for i in range(1, n):
            r[i][1] = yCap(*r[i - 1], *r[i])

        for i in range(n - 2, -1, -1):
            r[i][1] = yCap(*r[i + 1], *r[i])

        res = 0
        for i in range(1, n):
            res = max(res, yPeak(*r[i - 1], *r[i]))

        return max(res, r[-1][1] + num - r[-1][0])


class Solution:
    def maxIceCream(self, costs: List[int], coins: int) -> int:
        xMin = xMax = max(costs)
        freq = [0] * (xMax + 1)
        for x in costs:
            freq[x] += 1
            xMin = min(xMin, x)
        cnt = 0
        for x, f in enumerate(freq[xMin:], start=xMin):
            if f == 0:
                continue
            buy = min(coins // x, f)
            if buy == 0:
                break
            cnt += buy
            coins -= buy * x
        return cnt

class Solution:
    def maxNumberOfBalloons(self, text: str) -> int:
        # Create a frequency map for the input string
        freqMap = {}
        for c in text:
            freqMap[c] = freqMap.get(c, 0) + 1

        # Store the required frequencies of each character in 'balloon'
        balloon = "balloon"

        # Calculate the max number of "balloon" words
        maxBalloons = float('inf')
        balloonFreq = {}
        for c in balloon:
            balloonFreq[c] = balloonFreq.get(c, 0) + 1

        # Calculate the max possible number of "balloon" we can form
        for key, count in balloonFreq.items():
            maxBalloons = min(maxBalloons, freqMap.get(key, 0) // count)

        return 0 if maxBalloons == float('inf') else maxBalloons


class Solution:
    def zigZagArrays(self, n: int, l: int, r: int) -> int:
        MOD = 1000000007
        m = r - l + 1
        dp = [1] * m

        for i in range(2, n + 1):
            dp.reverse()
            s = 0
            for j in range(m):
                dp[j], s = s, (s + dp[j]) % MOD

        return (sum(dp) % MOD << 1) % MOD


class Solution:
    def zigZagArrays(self, n: int, l: int, r: int) -> int:
        MOD = 1000000007

        m = r - l + 1
        sz = 2 * m

        def multiply(A, B):
            C = [[0] * sz for _ in range(sz)]

            for i in range(sz):
                for k in range(sz):
                    if A[i][k] == 0:
                        continue

                    cur = A[i][k]

                    for j in range(sz):
                        if B[k][j] == 0:
                            continue

                        C[i][j] = (C[i][j] + cur * B[k][j]) % MOD

            return C

        T = [[0] * sz for _ in range(sz)]

        for x in range(m):

            for y in range(x + 1, m):
                T[x][m + y] = 1

            for y in range(x):
                T[m + x][y] = 1

        result = [[0] * sz for _ in range(sz)]
        for i in range(sz):
            result[i][i] = 1

        power = n - 1

        while power:
            if power & 1:
                result = multiply(result, T)

            T = multiply(T, T)
            power >>= 1

        answer = 0

        for i in range(sz):
            row_sum = sum(result[i]) % MOD
            answer = (answer + row_sum) % MOD

        return answer


class Solution:
    def countMajoritySubarrays(self, nums: List[int], target: int) -> int:
        n = len(nums)
        ans = 0

        for l in range(n):
            target_count = 0

            for r in range(l, n):
                if nums[r] == target:
                    target_count += 1

                length = r - l + 1

                if target_count > length // 2:
                    ans += 1

        return ans

class Solution:
    def countMajoritySubarrays(self, nums, target):
        n = len(nums)
        cnt = 0

        for i in range(n):
            freq = 0

            for j in range(i, n):
                if nums[j] == target:
                    freq += 1

                length = j - i + 1

                if freq > length // 2:
                    cnt += 1

        return cnt


class Solution:
    def maximumLength(self, nums: list[int]) -> int:
        freq = Counter(nums)
        res = (freq.pop(1, 0) - 1) | 1

        for f in freq:
            x = f
            sq = isqrt(x)
            if sq * sq == x and freq.get(sq, 0) > 1:
                continue

            n = 0
            while x < 31623 and freq.get(x, 0) > 1:
                n += 2
                x *= x

            res = max(res, n + ((x in freq) << 1) - 1)

        return res

class Solution:
    def maximumElementAfterDecrementingAndRearranging(self, A: list[int]) -> int:
        A.sort()
        n = len(A)

        A[0] = 1
        for i in range(1, n):
            A[i] = min(A[i], A[i - 1] + 1)
            
        return A[-1]

class Solution:
    def numOfStrings(self, patterns: list[str], word: str) -> int:
        count = 0
        for s in patterns:
            if word.find(s) != -1: # return -1 when not found
                count += 1
        return count