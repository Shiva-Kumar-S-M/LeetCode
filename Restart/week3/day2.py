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