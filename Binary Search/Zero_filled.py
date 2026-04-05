from collections import defaultdict
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