import json
import os
from datetime import datetime


class BaseExperiment:
    class Config:
        _obj = False
        
        def __init__(self):
            self._obj = True
            
    def __init__(self):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def _attrs(config):
        descriptions = {}

        for attr in dir(config):
            if attr != 'attrs' and attr != '_obj' and not attr.startswith('__'):
                obj = getattr(config, attr)
                
                if any([isinstance(obj, t) for t in [str, bool, int, float, list, dict]]):
                    descriptions[attr] = getattr(config, attr)
                
                elif getattr(obj, '_obj', False) and len(BaseExperiment._attrs(obj)):
                    descriptions[attr] = BaseExperiment._attrs(obj)

        return descriptions
    
    def __str__(self):
        return json.dumps(BaseExperiment._attrs(self), indent=4)

    @staticmethod
    def prepare(*paths):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
