class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s)!=len(t): return False

        ls=set(s)
        for l in ls:
            if s.count(l)!=t.count(l):
                return False
        return True
        