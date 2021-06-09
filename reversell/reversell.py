from typing import Optional, Union, Any, Tuple

class Node:
    def __init__(self, item: Any, _next: Optional['Node'] = None):
        self._next: Optional['Node'] = _next
        self.item: Any = item  

    def __str__(self):
        return f"({self.item})"


class LinkedList:
    def __init__(self):
        self.head: Optional[Node] = None
        self.n: int = 0

    def __str__(self):
        if self.head is None:
            return "()"

        node = self.head

        string = ""
        string += str(node)

        while node := node._next:
            string += f"->{node}"

        return string

    def append(self, item: Any): 
        if self.head is None:
            self.head = Node(item, None)
            return    

        node = self.head
        while node._next:
            node = node._next

        node._next = Node(item, None)
        self.n += 1


    def __iter__(self):
        if self.head is None:
            return
        
        node = self.head
        yield node

        while node := node._next:
            yield node

    def __len__(self):
        return self.n

    def __getitem__(self, index: int):
        if index > len(self) or index < 0:
            raise IndexError()
        
        node = self.head
        for i in range(index):
            node = node._next        

        return node.item

    def reverse_recursion(self):
        if self.n <= 1:
            return

        def _reverse(node: Node, prev: Node):  
            if node._next is None:
                node._next = prev
                return node                

            tail = _reverse(node._next, node)
            node._next = prev 
            return tail
        
        self.head = _reverse(self.head, None)

    def reverse_loop(self):
        if self.n <= 1:
            return

        lleft, left, right = None, self.head, self.head._next
        
        while True:
            left._next = lleft   
            lleft = left
            if right is None:      
                break
            left = right
            right = right._next

        self.head = left


l = LinkedList()
for i in range(1,9):
    l.append(i)
l.reverse_loop()
print(l)
l.reverse_recursion()
print(l)

for i in l:
    print(i)

print(l[7])
