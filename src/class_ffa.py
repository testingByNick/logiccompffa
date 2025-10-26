class FFA:
    def __init__(self, model, features, n_feat):
        self.model = model
        self.features = features
        self.n_feat = n_feat

    def compute(self, x, target, eps=0.1, samples=100):
        pred_orig = self.model.predict_proba([x])[0, target]
        scores = np.zeros(self.n_feat)
        
        for i in range(self.n_feat):
            total = 0
            for _ in range(samples):
                x_mod = x.copy()
                change = eps * (2 * np.random.random() - 1)
                x_mod[i] = np.clip(x_mod[i] + change, 0, 1)
                pred_mod = self.model.predict_proba([x_mod])[0, target]
                total += abs(pred_orig - pred_mod)
            scores[i] = total / samples
        
        if np.sum(scores) > 0:
            scores = scores / np.sum(scores)
        return scores