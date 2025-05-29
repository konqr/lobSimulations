import numpy as np
class ReplayBuffer:
    def __init__(self, obs_shape, action_shape, capacity=10000):
        self.buffer=[None]*capacity
        self.weights=np.zeros(capacity)
        self.count=0
        self.capacity=capacity
        self.max_weight=0.001
        self.indices=None
        self.delta=0.0001
        self.pointer=0
    
    def add_sample(self, transition):
        self.buffer[self.pointer]=transition
        self.weights[self.pointer]=self.max_weight
        self.pointer=(self.pointer+1)%self.capacity
        self.count=min(self.count+1, self.capacity)


    def sample_minibatch(self, batchsize=100):
        ws=self.weights[:self.count]+self.delta
        probabilities=ws/sum(ws)
        self.indices=np.random.choice(range(self.count), batchsize, p=probabilities, replace=False)
        return [self.buffer[i] for i in self.indices]
    
    def update_weights(self, errors):
        max_error=max(errors)
        self.max_weight=max(self.max_weight, max_error)
        self.weights[self.indices]=errors
