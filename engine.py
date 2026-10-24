
class Value:

    def __init__(self,data,_children=(),_op=''):
        self.data = data
        self.grad = 0 
        self._backward = lambda: None
        self._prev = set(_children)
        self.op = _op

    def __add__(self,other):
    
        other = other if isinstance(other,Value) else Value(other)
            #this means if we dont have other assign as value 
            # direct have the object then it also run eg. a + 7....
        out = Value(self.data + other.data, (self,other), '+')

        def _backward():
            self.grad+= out.grad
            other.grad+= out.grad
        out._backward = _backward
        # here backward is for back proporgation when we use that
        #  in micrograd it use this from out to self to other
        return out

    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

            out._backward = _backward

        return out

    
        # += instead of = cause if we dont use +=v then a bug in our 
        #  function is that if 1 value is using multiple times 
        # then this goes wrong as in grad cause it calculate single
        #  time the value so our grad will bie wrong
        # so thats why we are using += instead of =... 


    def __pow__ (self , other ): # for the function like exp and tanh,sinh,etc...
         assert isinstance(other, (int, float)), "only supporting int/float powers for now"
         out = Value(self.data**other, (self,), f'**{other}')

         def _backward():
            self.grad += (other*self.data** (other-1)) * out.grad
         out._backward = _backward

         return out
    

    def relu(self):
        out = Value(0 if self.data<0 else self.data , (self,),'Relu')

        def _backward():
            self.grad += (out.data>0) * out.grad
        out._backward = _backward

        return out
          

    def backward (self):
        self.grad = 1

        topo = []
        #topo = correct backward order
        visited = set()
        #visited is for that we dont reach the node more than 1 time
        def build_topo (v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                     build_topo(child)
                #this is recursion and give the surity that we reached every single node
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()
            
    def __neg__(self): #mean self is negative
        return (self) * -1 
    
    def __radd__(self , other): # mean other + self
        return self + other

    def __sub__(self,other):
        return self + (-other)

    def __rsub__ (self , other): # mean other - self
        return other + (-self)
    
    def __rmul__ (self , other): # mean other * self
        return self * other

    def __truediv__ (self , other):
        return self * other**-1

    def __rtruediv__ (self , other):
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})" # here f is the fstring..


        


