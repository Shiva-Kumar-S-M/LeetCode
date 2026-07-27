class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        res=""

        v=sorted(strs)
        left=v[0]
        right=v[-1]

        for i in range(min(len(left),len(right))):
            if left[i]!=right[i]:
                return res
            res+=left[i]
        return res
        