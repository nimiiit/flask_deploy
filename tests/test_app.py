import utils
import torch

def test_prediction():
    image=torch.rand([1,3,64,64])
    prediction =utils.get_prediction(image)
    assert len(prediction)==1