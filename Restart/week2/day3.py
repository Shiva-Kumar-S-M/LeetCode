#88 Merge sorted array

from typing import List


class Solution:
    def merge(self,nums1,m,nums2,n):
        p1,p2,p=m-1,n-1,m+n-1

        while p1>=0 and p2>=0:
            if nums1[p1]>nums2[p2]:
                nums1[p]=nums1[p1]
                p1-=1
            else:
                nums1[p]=nums2[p2]
                p2-=1
            p-=1

        while p2>=0:
            nums1[p]=nums2[p2]
            p-=1
            p2-=1


class Solution:
    def merge(self,nums1,m,nums2,n):
        for i in range(n):
            nums1[m+i]=nums2[i]
        nums1.sort()



#Two sum 2 input array is sorted
class Solution:
    def twoSum(self,numbers,target):
        left,right=0,len(numbers)-1

        while left<right:
            sum=numbers[left]+numbers[right]

            if sum==target:
                return [left+1,right+1]
            
            elif sum<target:
                left+=1
            else:
                right-=1
        return []
    

#643 Maximum Average SubArray using sliding Window (easy)

class Solution:
    def findMaxAverage(self,nums,k):
        currentSum=sum(nums[:k])
        maxSum=currentSum

        for i in range(k,len(nums)):
            currentSum=currentSum-nums[i-k]+nums[i]
            maxSum=max(maxSum,currentSum)

        return maxSum/k
    

#438 Find all anagrams in a string using sliding window (Medium)

class Solution:
    def findAnagrams(self,s,p):
        res=[]
        if len(s)<len(p):
            return res
        pCount=[0]*26
        sCount=[0]*26

        for i in range(len(p)):
            sCount[ord(s[i])-ord('a')]+=1
            pCount[ord(p[i])-ord('a')]+=1


        for i in range(0,len(s)-len(p)+1):
            if sCount==pCount:
                res.append(i)

            if i+len(p)<len(s):
                sCount[ord(s[i])-ord('a')]-=1
                sCount[ord(s[i+len(p)])-ord('a')]+=1

        return res
    
#567 Permutation in string using Sliding window (Medium)
class Solution:
    def CheckInclusion(self,s1,s2):
        if len(s1)>len(s2):
            return False
        
        s1Count=[0]*26
        s2Count=[0]*26

        for i in range(len(s1)):
            s1Count[ord(s1[i])-ord('a')]+=1
            s2Count[ord(s2[i])-ord('a')]+=1

        for i in range(len(s2)-len(s1)+1):
            if self.matches(s1Count,s2Count):
                return True
            
            s2Count[ord(s2[i])-ord('a')]-=1
            s2Count[ord(s2[i+len(s1)])-ord('a')]+=1

        return self.matches(s1Count,s2Count)
    
    def matches(self,a,b):
        for i in range(26):
            if a[i]!=b[i]:
                return False
        return True
    
#2461. Maximum Sum of Distinct Subarrays With Length K Using sliding window (medium)

class Sloution:
    def maximumSubarray(self,nums,k):
        n=len(nums)
        seen=set()
        left,right,s,maxSum=0,0,0,0

        while right<n:
            while nums[right] in seen:
                seen.remove(nums[left])
                s-=nums[left]
                left+=1

            seen.add(nums[right])
            s+=nums[right]
            right+=1

            if right-left==k:
                maxSum=max(s,maxSum)
                seen.remove(nums[left])
                s-=nums[left]
                left+=1

        return maxSum
    
class SegmentTree:
    """Segment Tree over array of size n"""

    def __init__(self, n: int):
        self.n = n
        self.size = 4 * n
        self.sum = [0] * self.size
        self.min = [0] * self.size
        self.max = [0] * self.size

    def _pull(self, node: int):
        """Helper to recompute information of node by it's children"""

        l, r = node * 2, node * 2 + 1

        self.sum[node] = self.sum[l] + self.sum[r]
        self.min[node] = min(self.min[l], self.sum[l] + self.min[r])
        self.max[node] = max(self.max[l], self.sum[l] + self.max[r])

    def update(self, idx: int, val: int):
        """Update value by index idx in original array"""

        node, l, r = 1, 0, self.n - 1
        path = []

        while l != r:
            path.append(node)
            m = l + (r - l) // 2
            if idx <= m:
                node = node * 2
                r = m
            else:
                node = node * 2 + 1
                l = m + 1

        self.sum[node] = val
        self.min[node] = val
        self.max[node] = val

        while path:
            self._pull(path.pop())

    def find_rightmost_prefix(self, target: int = 0) -> int:
        """Find rightmost index r with prefixsum(r) = target
        prefixsum(i) = sum(arr[j] for j in range(i + 1))"""

        node, l, r, sum_before = 1, 0, self.n - 1, 0

        def _exist(node: int, sum_before: int):
            return self.min[node] <= target - sum_before <= self.max[node]

        if not _exist(node, sum_before):
            return -1

        while l != r:
            m = l + (r - l) // 2
            lchild, rchild = node * 2, node * 2 + 1

            # Check right half first
            sum_before_right = self.sum[lchild] + sum_before
            if _exist(rchild, sum_before_right):
                node = rchild
                l = m + 1
                sum_before = sum_before_right
            else:
                node = lchild
                r = m

        return l


class Solution:
    def longestBalanced(self, nums: List[int]) -> int:
        n = len(nums)

        stree = SegmentTree(n)  # SegmentTree over balance array for current l
        first = dict()  # val -> first occurence idx for current l

        result = 0
        for l in reversed(range(n)):
            num = nums[l]
    
            # If x already had a first occurrence to the right, remove that old marker.
            if num in first:
                stree.update(first[num], 0)

            # Now x becomes first occurrence at l.
            first[num] = l
            stree.update(l, 1 if num % 2 == 0 else -1)

            # Find rightmost r >= l such that sum(w[l..r]) == 0
            r = stree.find_rightmost_prefix(target=0)
            if r >= l:
                result = max(result, r - l + 1)

        return result