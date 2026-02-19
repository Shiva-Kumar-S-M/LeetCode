#Remove nth node from the end of linked list
from typing import List


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def removenthfromend(self,head,n):
        dummy=ListNode(0,head)
        first=dummy
        second=dummy

        for i in range(n+1):
            first=first.next

        while first:
            first=first.next
            second=second.next

        second.next=second.next.next

        return dummy.next
    
# 82 remove duplicates from sorted list 2 (medium)

class Solution:
    def reomveDuplicates(self,head):
        dummy=ListNode(0,head)
        prev=dummy
        cur=head

        while cur is not None:
            while cur.next is not None and cur.val==cur.next.val:
                cur=cur.next

            if prev.next is cur:
                prev=prev.next
            else:
                prev.next=cur.next
            
            cur=cur.next

        return dummy.next
    
#83 remove duplicates from sorted array(easy)
class Solution:
    def removeDuplicates(self,head):
        cur=head

        while head and head.next:
            if head.val==head.next.val:
                head.next=head.next.next
            else:
                head=head.next
        return cur
    
#401 Binary Watch(easy)
class Solution:
    def readBinaryWatch(self, k: int) -> List[str]:
        if k == 0:
            return ['0:00']
        mask = (1 << 6) - 1
        q = (1 << k) - 1
        limit = q << (10 - k)
        res = []
        while q <= limit:
            min = q & mask
            hour = q >> 6
            if hour < 12 and min < 60:
                res.append(f'{hour}:{min:0>2}')
            r = q & -q
            n = q + r 
            q = (((q ^ n) // r) >> 2) | n
        return res


class Solution:
    def readBinaryWatch(self,n):
        x=x^(x>>1)
        return x&(x+1)==0


class Solution:
    def countBinarySubstrings(self, s: str) -> int:
        res = 0
        prev = 0
        strk = 1

        for i in range(1, len(s)):
            if s[i] == s[i - 1]: strk += 1
            else:
                prev = strk
                strk = 1

            if strk <= prev: res += 1

        return res
