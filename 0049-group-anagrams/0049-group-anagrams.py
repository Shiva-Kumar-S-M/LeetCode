class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res=defaultdict(list)
        

        for i in strs:
            key="".join(sorted(i))
            res[key].append(i)
        return list(res.values())