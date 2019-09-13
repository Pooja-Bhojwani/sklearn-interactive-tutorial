class BinarizeTransformer(TransformerMixin):
    
    def __init__(self, threshold = 0):
        self.threshold = threshold
        
    def fit(self, x, y = None):
        return self
    
    def transform(self, x):
        cond = x > self.threshold
        not_cond = np.logical_not(cond)
        x[cond] = 1
        x[not_cond] = 0
        return x

