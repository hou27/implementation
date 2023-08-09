def update_params(params, grads, learning_rate=1e-3):
    for param, grad in zip(params, grads):
        param -= learning_rate * grad
    
    return params