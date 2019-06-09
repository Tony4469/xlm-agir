import requests
import urllib 

def count_words_at_url(url):
    print('counting words')
    resp = requests.get(url)
    print('counting ok')
    
    url = "https://dl.fbaipublicfiles.com/XLM/mlm_tlm_xnli15_1024.pth"
    urllib.request.urlretrieve(url, "mlm_tlm_xnli15_1024.pth")
 
    print('file dwnld ok')
    return len(resp.text.split())