#moving zeros 283
class Solution:
    def movZeros(self,nums):
        list=0
        for i in range(len(nums)):
            if nums[i]!=0:
                nums[list],nums[i]=nums[i],nums[list]
                list+=1
        return list
    

#majority elements 169
class Solution:
    def majorityElement(self,nums):
        candidate=nums[0]
        count=0
        for num in nums:
            if count==0:
                candidate=num
        count+=1 if num==candidate else -1
        return candidate
    
#remove duplicates from sorted array 26
class Solution:
    def removeDuplicates(self,nums):
        if not nums:
            return 0
        res=0
        for i in range(1,len(nums)):
            if nums[i]!=nums[res]:
                res+=1
                nums[res]=nums[i]
        return res+1
    


#longest common prefix 14
class Solution:
    def LongestCommonPrefix(self,strs):
        res=""
        a=sorted(strs)
        first=a[0]
        last=a[-1]

        for i in range(min(len(first),len(last))):
            if first[i]!=last[i]:
                return res
            res+=first[i]
        return res

#rotate array 189

class Solution:
    def rotate(self,nums,k):
        n=len(nums)
        k%=n

        def reverse(l,r):
            while l<r:
                nums[l],nums[r]=nums[r],nums[l]
                l+=1
                r-=1

        reverse(0,n-1)
        reverse(0,k-1)
        reverse(k,n-1)


#Balance Binary search tree 
class Solution:
    def inorder(self, node, vals):
        if not node:
            return
        self.inorder(node.left, vals)
        vals.append(node.val)
        self.inorder(node.right, vals)

    def build(self, vals, l, r):
        if l > r:
            return None
        mid  = (l + r) // 2
        node = TreeNode(vals[mid]) # type: ignore
        node.left  = self.build(vals, l, mid - 1)
        node.right = self.build(vals, mid + 1, r)
        return node

    def balanceBST(self, root):
        vals = []
        self.inorder(root, vals)
        return self.build(vals, 0, len(vals) - 1)