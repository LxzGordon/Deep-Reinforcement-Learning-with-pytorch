class replay_memory():
    def __init__(self,replay_memory_size):
        self.memory_size=replay_memory_size
        self.memory=np.array([])
        self.cur=0
        self.new=0
    def size(self):
        return self.memory.shape[0]
#[s,a,r,s_,done] make sure all info are lists, i.e. [[[1,2],[3]],[1],[0],[[4,5],[6]],[True]]
    def store_transition(self,trans):
        if(self.memory.shape[0]<self.memory_size):
            if self.new==0:
                self.memory=np.array(trans)
                self.new=1
            elif self.memory.shape[0]>0:
                self.memory=np.vstack((self.memory,trans))
        else:
            self.memory[self.cur,:]=trans
            self.cur=(self.cur+1)%self.memory_size
    
    def sample(self):
        if self.memory.shape[0]<batch_size:
            return -1
        sam=np.random.choice(self.memory.shape[0],batch_size)
        return self.memory[sam]
