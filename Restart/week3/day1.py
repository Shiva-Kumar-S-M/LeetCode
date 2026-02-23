#21 Merge two sorted lists (easy) using recursion

class Solution:
    def mergeTwoLists(self,l1,l2):
        if l1 is None:
            return l2
        if l2 is None:
            return l1
        
        if l1.val<=l2.val:
            l1.next=self.mergeTwoLists(l1.next,l2)
            return l1
        else:
            l2.next=self.mergeTwoLists(l1,l2.next)
            return l2
        

#Pow(x,n) (medium) using recursion
class Solution:
    def myPow(self,x,n):
        N=n 
        if N<0:
            x=1/x
            N=-N

        res=1
        cur=x

        while N>0:
            if N%2==1:
                res*=cur
            cur*=cur
            N//=2

        return res
    

#1461. Check If a String Contains All Binary Codes of Size K (medium) using set
class Solution:
    def hasAllCodes(self,s,k):
        if len(s)<k:
            return False
        
        res=set()
        target=1<<k

        for i in range(len(s)-k+1):
            res.add(s[i:i+k])
        return len(res)==target
    
