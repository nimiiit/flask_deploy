import requests

resp = requests.post(" https://genderpredictiontest123.herokuapp.com/predict", files={'file':open('Tiny_Portrait_000000.png', 'rb')})

print(resp.text)
