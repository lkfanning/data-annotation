"""
Learning Rate Decay Strategies

This script demonstrates different learning rate decay strategies that can be used
in the logistic_softmax_train.py file. You can experiment with these strategies
to find the one that works best for your specific problem.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
initial_learning_rate = 0.001  # Starting learning rate
max_iterations = 10000        # Total number of training iterations

# Different decay strategies

def exponential_decay(initial_lr, decay_rate=0.9, decay_steps=1000):
    """
    Exponential decay: lr = initial_lr * decay_rate^(step/decay_steps)
    
    Args:
        initial_lr: Starting learning rate
        decay_rate: Rate at which learning rate decreases (0 to 1)
        decay_steps: Number of steps after which to apply decay
    """
    lr_values = []
    for i in range(max_iterations):
        lr = initial_lr * (decay_rate ** (i / decay_steps))
        lr_values.append(lr)
    return lr_values

def step_decay(initial_lr, drop_rate=0.5, epochs_drop=1000):
    """
    Step decay: lr = initial_lr * drop_rate^floor(epoch/epochs_drop)
    
    Args:
        initial_lr: Starting learning rate
        drop_rate: Factor by which to reduce learning rate (0 to 1)
        epochs_drop: Number of epochs after which to drop learning rate
    """
    lr_values = []
    for i in range(max_iterations):
        lr = initial_lr * (drop_rate ** (i // epochs_drop))
        lr_values.append(lr)
    return lr_values

def polynomial_decay(initial_lr, power=1.0, min_lr=0.0001):
    """
    Polynomial decay: lr = (initial_lr - min_lr) * (1 - step/max_iterations)^power + min_lr
    
    Args:
        initial_lr: Starting learning rate
        power: Power of decay function (higher = faster decay at start)
        min_lr: Minimum learning rate value
    """
    lr_values = []
    for i in range(max_iterations):
        progress = min(1.0, i / max_iterations)
        lr = (initial_lr - min_lr) * ((1 - progress) ** power) + min_lr
        lr_values.append(lr)
    return lr_values

def cosine_decay(initial_lr, min_lr=0.0):
    """
    Cosine annealing: lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(π * step/max_iterations))
    
    Args:
        initial_lr: Starting learning rate
        min_lr: Minimum learning rate value
    """
    lr_values = []
    for i in range(max_iterations):
        progress = min(1.0, i / max_iterations)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        lr = min_lr + (initial_lr - min_lr) * cosine_decay
        lr_values.append(lr)
    return lr_values

# Plot all strategies
plt.figure(figsize=(12, 8))

exp_decay_values = exponential_decay(initial_learning_rate)
step_decay_values = step_decay(initial_learning_rate)
poly_decay_values = polynomial_decay(initial_learning_rate)
cosine_decay_values = cosine_decay(initial_learning_rate)

plt.plot(exp_decay_values, label='Exponential Decay')
plt.plot(step_decay_values, label='Step Decay')
plt.plot(poly_decay_values, label='Polynomial Decay')
plt.plot(cosine_decay_values, label='Cosine Decay')

plt.title('Learning Rate Decay Strategies')
plt.xlabel('Iterations')
plt.ylabel('Learning Rate')
plt.legend()
plt.grid(True)
plt.savefig('/tmp/outputs/learning_rate_decay_strategies.png')
plt.show()

# How to use in logistic_softmax_train.py:
print("IMPLEMENTATION GUIDE:")
print("================================================")
print("1. EXPONENTIAL DECAY (used in updated script):")
print("   learning_rate = initial_learning_rate * (decay_rate ** (i / decay_steps))")
print("   - Smoothly decreases learning rate")
print("   - Recommended parameters: decay_rate=0.9, decay_steps=1000")
print()
print("2. STEP DECAY:")
print("   learning_rate = initial_learning_rate * (drop_rate ** (i // epochs_drop))")
print("   - Decreases learning rate in discrete steps")
print("   - Recommended parameters: drop_rate=0.5, epochs_drop=1000")
print()
print("3. POLYNOMIAL DECAY:")
print("   progress = min(1.0, i / max_iterations)")
print("   learning_rate = (initial_lr - min_lr) * ((1 - progress) ** power) + min_lr")
print("   - Gradual decay with controllable curvature")
print("   - Recommended parameters: power=1.0, min_lr=0.0001")
print()
print("4. COSINE DECAY:")
print("   progress = min(1.0, i / max_iterations)")
print("   cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))")
print("   learning_rate = min_lr + (initial_lr - min_lr) * cosine_decay")
print("   - Smooth decay with slower decrease in middle of training")
print("   - Recommended parameters: min_lr=0.0")
print("================================================")
