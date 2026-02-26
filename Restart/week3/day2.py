#Reversed linked list (easy)

class ListNode:
    def __init__(self,val=0,next=None):
        self.val=val
        self.next=next

class Solution:
    def reverseList(self,head):
        prev=None
        cur=head

        while cur is not None:
            next=cur.next
            cur.next=prev
            prev=cur
            cur=next

        return prev
    
class Solution:
    def sumRootToLeaf(self, root: TreeNode) -> int: # type: ignore

        def dfs(node: TreeNode, n = 0)-> None: # type: ignore
            if not node: return

            n = 2 * n + node.val
            if not node.left and not node.right:
                self.ans+= n
                return
                
            dfs(node.left , n)
            dfs(node.right, n)
            return
            

        self.ans = 0
        dfs(root)
        return self.ans

class Solution:
    def sortByBits(self, arr):
        return sorted(arr, key=lambda x: (x.bit_count(), x))


class Solution:
    def numSteps(self, s: str) -> int:
        steps = 0
        carry = 0
        for i in range(len(s) - 1, 0, -1):
            bit = ord(s[i]) & 1
            steps += 1 + (bit ^ carry)
            carry |= bit

        return steps + carry