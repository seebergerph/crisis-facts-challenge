import src.registry as registry


@registry.register("dummy-reranker")
class DummyReranker():
    def __init__(self):
        pass

    def score(self, initial_results):
        # do nothing
        return initial_results

    def setup(self):
        pass

    def clear(self):
        pass