class AdderTransformer(TransformerMixin):
    
    def __init__(self, add=0):
        self.add = add
        
    def fit(self, x, y = None):
        return self
    
    def transform(self, x):
        return x + self.add
    
class MeanNormalizer(TransformerMixin):
    
    def __init__(self):
        pass
        
    def fit(self, x, y = None):
        self.means = x.mean(axis=0)
        return self
    
    def transform(self, x):
        return x - self.means    
    
class TransformerPipeline(TransformerMixin):
    
    def __init__(self, transformers):
        self.transformers = transformers
        
    def fit(self, x, y = None):
        x_ = x.copy()
        for transformer in self.transformers:
            transformer.fit(x_)
        return self
        
    def transform(self, x):
        x_ = x.copy()
        for transformer in self.transformers:
            x_ = transformer.transform(x_)
        return x_