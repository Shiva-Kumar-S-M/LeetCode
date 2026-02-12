#11 Container with most water using two pointer approcah (Medium)
class Solution:
    def maxArea(self,height):
        left=0
        right=len(height)-1
        maxArea=0

        while left<right:
            area=min(height[left],height[right])*(right-left)
            maxArea=max(area,maxArea)

            if height[left]<height[right]:
                left+=1
            else:
                right-=1

        return maxArea
    
#3 Longest Substring Without Repeating Characters (Medium)
class Solution:
    def lengthOfLongestSubstring(self,s): 
        charSet=set() 
        left=0 
        maxLength=0  

        for right in range(len(s)):
            while s[right] in charSet:
                charSet.remove(s[left])
                left+=1
            charSet.add(s[right])
            maxLength=max(maxLength,right-left+1)
        return maxLength    
    
#424 Longest Repeating Character Replacement (Medium)
class Solution:
     def characterReplacement(self,s,k): 
        charCount={} 
        left=0 
        maxCount=0 
        maxLength=0 
        for right in range(len(s)): 
            charCount[s[right]]=charCount.get(s[right],0)+1
            maxCount=max(maxCount,charCount[s[right]])
            if (right-left+1)-maxCount>k:
                charCount[s[left]]-=1
                left+=1
            maxLength=max(maxLength,right-left+1)
        return maxLength
     
class Solution:
    def longestBalanced(self, s: str) -> int:
        cnt, n=1, len(s)
        for l in range(n):
            freq=[0]*26
            uniq, maxF, cntMax=0, 0, 0
            for r in range(l, n):
                freq[ord(s[r])-97]+=1
                f=freq[ord(s[r])-97]
                uniq+=f==1
                if f>maxF:
                    maxF=f
                    cntMax=1
                elif f==maxF:
                    cntMax+=1
                if uniq==cntMax:
                    cnt=max(cnt, r-l+1)
        return cnt