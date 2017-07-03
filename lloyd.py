import numpy as np

#lloyds algorithm

#iteratively minimize sum(k=1,k){sum(xn elt Sk) {||xn-mk||^2}} wrt mk,Sk
#where mk is the mth center and Sk is the kth cluster

#mk = (1/|Sk|) sum(xn elt Sk){xn}
#Sk = {xn : ||xn - mk|| <= all ||xn - ml||}


class Lloyd:
    def calc_cluster_centers(self):
        #return true if all clusters nonempty, else false
        nonempty = True # if all clusters nonempty 
        for k,cluster in enumerate(self.cluster):
            if len(cluster) <= 0:
                nonempty = False
                break
            else:
                cur_cluster = self.X[cluster]
                self.cluster_center[k] = np.average(cur_cluster, axis=0)
        return nonempty
            
            
    def assign_clusters(self):
        #returns if cluster membership changed or not
        changed = False
        #hopefully X is two-dim or else this breaks
        #iterate over X
        for n, xn in enumerate(self.X):
            cur_cluster = self.X_cluster[n]
            dest_cluster = cur_cluster #cluster that current xn ends up in
            shortest_dist = np.linalg.norm(self.cluster_center[cur_cluster]-xn) #dist of xn from current cluster
            #iterate over clusters
            for l, cluster in enumerate(self.cluster_center):
                cur_dist = np.linalg.norm(cluster - xn) #dist of xn from iterated cluster
                if cur_dist < shortest_dist:
                    dest_cluster = l
                    shortest_dist = cur_dist
            if cur_cluster != dest_cluster:
                self.cluster[cur_cluster].remove(n)
                self.cluster[dest_cluster].append(n)
                self.X_cluster[n] = dest_cluster
                changed = True
        return changed
                
    
    def init_clusters(self):
        self.cluster = [[] for x in range(self.k)]
        self.cluster[0] = [x for x in range(self.X_n)] #stick all in the first cluster for now
        #listing of cluster membership by elts of X
        self.X_cluster = [0 for x in range(self.X_n)]
        #cluster centers
        self.cluster_center = np.random.uniform(self.rng[0], self.rng[1], (self.k, self.X_dim))
        self.assign_clusters()

    def set_X(self,X):
        #X = dataset, should be m x n np array
        self.X = X
        self.X_n = X.shape[0]
        if len(X.shape) == 1:
            self.X_dim = 1
        else:
            self.X_dim = X.shape[1]
        self.init_clusters()


    def __init__(self, X, k, rng):
        #k = number of clusters
        self.k = max(1, int(k))
        #rng = range of allowed center coords as an array
        self.rng = rng
        self.set_X(X)


    def set_k(self,k):
        if k != self.k:
            self.k = max(1, int(k))
            self.init_clusters()

    def run(self):
        runs = 1 #number of runs executed
        while True:
            while True:
                nonempty = self.calc_cluster_centers()
                if nonempty == True:
                    break
                else:
                    self.init_clusters()
                    runs = runs + 1
            changed = self.assign_clusters()
            if changed == False:
                break
        return runs
                    
        

    
    
    
        
