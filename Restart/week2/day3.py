#88 Merge sorted array

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
    