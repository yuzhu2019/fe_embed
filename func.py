import pickle

def save_dict(obj, file_pkl):
    with open(file_pkl, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(file_pkl):
    with open(file_pkl, 'rb') as f:
        return pickle.load(f)
    
def save_list(obj, file_txt):
    with open(file_txt, 'wb') as f:
        pickle.dump(obj, f)    
        
def load_list(file_txt): 
    with open(file_txt, 'rb') as f:
        obj = pickle.load(f)
    return obj

  