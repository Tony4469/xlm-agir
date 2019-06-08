from six.moves import urllib

def initialize_data():
    """
    """
    print('downloading model')
    url = "https://dl.fbaipublicfiles.com/XLM/mlm_tlm_xnli15_1024.pth"
    urllib.request.urlretrieve(url, "mlm_tlm_xnli15_1024.pth")
    print('file downloaded')
    
    return True


print('initializing download')
if __name__ == '__init__':
    initialize_data()
















