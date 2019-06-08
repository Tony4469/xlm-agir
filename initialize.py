import requests
import io

class Reloaded():
    def __init__(self):
        super().__init__()
        print('RELOADED loaded')
    
    def serve(self):
        return reloaded

def princ():
    print('downloading model')
    url = "https://dl.fbaipublicfiles.com/XLM/mlm_tlm_xnli15_1024.pth"
#    urllib.request.urlretrieve(url, "mlm_tlm_xnli15_1024.pth")

    chemin = getcwd()
    curPath = chemin if "xlm" in chemin else (path.join(getcwd(), 'xlm'))

#    model_path = path.normpath(path.join(curPath, './mlm_tlm_xnli15_1024.pth') )
    
    print('allez le model')
    response = requests.get(url)
    print('response downloaded')
    f = io.BytesIO(response.content)
    reloaded = torch.load(f)
    print('file downloaded')
    return reloaded

print('initializing download')
reloaded = princ()
















