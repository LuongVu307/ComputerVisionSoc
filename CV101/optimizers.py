import numpy as np





class SGD:
    #TODO
    ...


class Momentum:
    def __init__(self, learning_rate=0.001, momentum=0.9, weight_decay=0.0):
        """
        Initializes the Momentum optimizer.
        
        Parameters:
        - learning_rate: The step size for parameter updates.
        - momentum: The momentum factor (default is 0.9).
        - weight_decay: L2 regularization factor (default is 0).
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = None  # To store the previous velocity

    def step(self, params, grads):
        """
        Performs a parameter update based on gradients and momentum.
        
        Parameters:
        - params: List or array of model parameters.
        - grads: List or array of gradients for the parameters.
        """
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]  # Initialize velocity for each parameter

        for i in range(len(params)):
            if self.weight_decay > 0:
                grads[i] += self.weight_decay * params[i]

            self.velocity[i] = self.momentum * self.velocity[i] + grads[i]

            params[i] -= self.learning_rate * self.velocity[i]

    def get_config(self):
        """
        Get the configuration of the Momentum optimizer.
        """
        return {
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay
        }

class RMSprop:
    def __init__(self, learning_rate=0.001, decay=0.9, epsilon=1e-8, weight_decay=0.0):
        """
        Initializes the RMSprop optimizer.
        
        Parameters:
        - learning_rate: The step size for parameter updates.
        - decay: The decay rate for the moving average of squared gradients.
        - epsilon: A small constant added to prevent division by zero.
        - weight_decay: L2 regularization factor (default is 0).
        """
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.cache = None  # To store the moving average of squared gradients

    def step(self, params, grads):
        """
        Performs a parameter update based on the gradients.

        Parameters:
        - params: List or array of model parameters.
        - grads: List or array of gradients for the parameters.
        """
        if self.cache is None:
            self.cache = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            if self.weight_decay > 0:
                grads[i] += self.weight_decay * params[i]

            self.cache[i] = self.decay * self.cache[i] + (1 - self.decay) * grads[i] ** 2

            params[i] -= self.learning_rate * grads[i] / (np.sqrt(self.cache[i]) + self.epsilon)

    
    def get_config(self):
        """
        Get the configuration of the RMSprop optimizer.
        """
        return {
            'learning_rate': self.learning_rate,
            'decay': self.decay,
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay
        }

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        """
        Initializes the Adam optimizer.
        
        Parameters:
        - learning_rate: The step size for parameter updates.
        - beta1: The exponential decay rate for the first moment estimate (default is 0.9).
        - beta2: The exponential decay rate for the second moment estimate (default is 0.999).
        - epsilon: A small constant added to prevent division by zero (default is 1e-8).
        - weight_decay: L2 regularization factor (default is 0).
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None 
        self.v = None
        self.t = 0

    def step(self, params, grads):
        """
        Performs a parameter update based on gradients and Adam's algorithm.
        
        Parameters:
        - params: List or array of model parameters.
        - grads: List or array of gradients for the parameters.
        """
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]  # Initialize first moment estimate (m)
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]  # Initialize second moment estimate (v)
        
        self.t += 1  # Increment the time step
        
        for i in range(len(params)):
            # Apply weight decay (L2 regularization)
            if self.weight_decay > 0:
                grads[i] += self.weight_decay * params[i]
            
            # Update the first moment estimate (m) and the second moment estimate (v)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update the parameters
            params[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def zero_grad(self):
        """
        Resets the gradients (typically done after each backward pass).
        """
        pass  # No specific implementation as gradients are handled externally.

    def get_config(self):
        """
        Get the configuration of the Adam optimizer.
        """
        return {
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay
        }
