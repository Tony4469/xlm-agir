import requests

def count_words_at_url(url):
    print('counting words')
    resp = requests.get(url)
    print('counting ok')
    return len(resp.text.split())