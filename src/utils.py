import dill
import os
import sys
from src.exception import CustomException

def SaveModel(filepath, obj):
    try:
        dir = os.path.dirname(filepath)

        os.makedirs(dir, exist_ok=True)

        with open(filepath, 'wb') as fileobj:
            dill.dump(obj, fileobj)
    
    except Exception as e:
        raise CustomException(e,sys)
