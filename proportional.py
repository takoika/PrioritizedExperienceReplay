import random
import sum_tree


class Experience:
    """ The class represents prioritized experience replay buffer.

    The class has functions: store samples, pick samples with
    probability in proportion to sample's priority, update
    each sample's priority, reset alpha.

    see https://arxiv.org/pdf/1511.05952.pdf .

    """
    
    def __init__(self, memory_size, batch_size, alpha):
        """ Prioritized experience replay buffer initialization.

        Parameters
        ----------
        memory_size : int
            sample size to be stored
        batch_size : int
            batch size to be selected by `select` method
        alpha: float
            exponent determine how much prioritization.
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        """
        self.tree = sum_tree.SumTree(memory_size)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.alpha = alpha

    def add(self, data, priority):
        """ Add new sample.

        Parameters
        ----------
        data : object
            new sample
        priority : float
            sample's priority
        """
        self.tree.add(data, priority**self.alpha)

    def select(self, beta):
        """ The method return samples randomly.

        Parameters
        ----------
        beta : float

        Returns
        -------
        out :
            list of samples
        weights:
            list of weight
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
        """

        if self.tree.filled_size() < self.batch_size:
            return None, None, None

        out = []
        indices = []
        weights = []
        priorities = []
        for _ in range(self.batch_size):
            random_number = random.random()
            data, priority, index = self.tree.find(random_number)
            priorities.append(priority)
            weights.append((1./self.memory_size/priority)**beta if priority > 1e-16 else 0)
            indices.append(index)
            out.append(data)
            self.priority_update([index], [0]) # To avoid duplicating


        self.priority_update(indices, priorities) # Revert priorities

        weights /= max(weights) # Normalize for stability

        return out, weights, indices

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        for i, priority in zip(indices, priorities):
            self.tree.val_update(i, priority**self.alpha)

    def reset_alpha(self, alpha):
        """ Reset a exponent alpha.

        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i)**-old_alpha for i in range(self.tree.filled_size())]
        self.priority_update(range(self.tree.filled_size()), priorities)
