from datasets import LinemodOcclusion

class Suite:

    def __init__(self, prefix, timed=False):

        self.ds = LinemodOcclusion(prefix)
        self.timed = timed

        # placeholder for result storage
        self.results = None
        self.frame_id = None
        self.object_id = None