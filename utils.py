from __future__ import division, print_function, absolute_import


def transform_reward(reward, done):
    """
    >>> transform_reward(reward=0.5, done=False)
    -0.1
    >>> transform_reward(reward=1, done=False)
    10.0
    >>> transform_reward(reward=0, done=True)
    -2.0
    """
    # penalise extra steps even if not terminal: in theory one
    # can make at most 4 steps from any place to any other place
    if reward >= 1:
        return 10.
    elif not done:
        return -0.1
    else:
        return -2.0


if __name__ == '__main__':
    import doctest
    doctest.testmod()
