from math import isqrt


class Solution:
    def maxOperations(self, s: str) -> int:
        ones = 0
        res = 0
        for i, c in enumerate(s):
            if c == '1':
                ones += 1
            elif i > 0 and s[i - 1] == '1':
                res += ones
        return res
    

class Solution:
    def rangeAddQueries(self, n: int, queries: list[list[int]]) -> list[list[int]]:
        diff = [[0] * n for _ in range(n)]
        
        for r1, c1, r2, c2 in queries:
            diff[r1][c1] += 1
            if r2 + 1 < n: diff[r2 + 1][c1] -= 1
            if c2 + 1 < n: diff[r1][c2 + 1] -= 1
            if r2 + 1 < n and c2 + 1 < n: diff[r2 + 1][c2 + 1] += 1
        
        for i in range(n):
            for j in range(n):
                above = diff[i - 1][j] if i > 0 else 0
                left = diff[i][j - 1] if j > 0 else 0
                diag = diff[i - 1][j - 1] if i > 0 and j > 0 else 0
                diff[i][j] += above + left - diag
                
        return diff


#problem 3234
class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)

        pref = [0] * (n + 1)
        for i, ch in enumerate(s):
            pref[i+1] = pref[i] + (ch == '1')

        Z = [i for i, ch in enumerate(s) if ch == '0']
        m = len(Z)

        ans = 0

        i = 0
        while i < n:
            if s[i] == '0':
                i += 1
                continue
            j = i
            while j < n and s[j] == '1':
                j += 1
            L = j - i
            ans += L * (L + 1) // 2
            i = j

        B = isqrt(n) + 2

        def ones(l, r):
            return pref[r+1] - pref[l]

        for a in range(m):
            Lmin = 0 if a == 0 else Z[a-1] + 1
            Lmax = Z[a]
            if Lmin > Lmax:
                continue

            for z in range(1, B + 1):
                b = a + z - 1
                if b >= m:
                    break

                Rmin = Z[b]
                Rmax = Z[b + 1] - 1 if b + 1 < m else n - 1
                if Rmin > Rmax:
                    continue

                need = z * z
                r = Rmin

                for l in range(Lmin, Lmax + 1):
                    if pref[Rmax + 1] - pref[l] < need:
                        continue
                    while r <= Rmax and ones(l, r) < need:
                        r += 1
                    if r > Rmax:
                        break
                    ans += (Rmax - r + 1)

        return ans