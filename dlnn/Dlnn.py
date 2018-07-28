class Dlnn(object):
    @staticmethod
    def config(*args, **kwargs):
        return Dlnn()

    def train(self, x, y):
        self._train(x, y)
        return self._evaluate(x, y)

    def _train(self, x, y):
        # TODO : Place Training Process Here
        pass

    def _evaluate(self, x, y):
        # TODO : Place Evaluation Process Here
        pass
