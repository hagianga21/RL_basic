config_dqn = {
    'state_dim': (4, 84, 84),
    'action_dim': 2,
    # Exploration
    'exploration_rate': 1,
    'exploration_rate_decay': 0.99999975,
    'exploration_rate_min': 0.1,
    #DQN
    'gamma': 0.9,
    'memory_capacity': 40000,
    'batch_size': 32,
    'target_update': 15000,
    'learning_rate': 0.00025,
    'device': "cpu",
}