import json
import numpy

class JsonNumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for handling numpy data types.
    """
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super().default(obj)