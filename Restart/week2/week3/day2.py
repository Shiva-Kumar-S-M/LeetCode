#316 remove dulpicate letters (medium)

class Solution:
    def removeDuplicates(self,s):
        frq={}
        for char in s:
            frq[char]=frq.get(char,0)+1

        stack=[]
        in_stack=set()

        for i in s:
            frq[i]-=1

            if i in in_stack:
                continue
            while stack and stack[-1]>i and frq[stack[-1]]>0:
                rmv=stack.pop()
                in_stack.remove(rmv)
            stack.append(i)
            in_stack.add(i)
        return ''.join(stack)
    


    #2390 removing stars from a string (medium)

    class Solution:
        def removeStars(self,s):
            stack=[]
            for i in s:
                if i=='*':
                    stack.pop()

                else:
                    stack.append(i)

            return ''.join(stack)
        
        