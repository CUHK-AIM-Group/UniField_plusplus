import unittest
import torch
from diffsynth.models.wan_video_dit import masked_block_sdpa

class AttentionTests(unittest.TestCase):
    def test_masked_output_and_gradients(self):
        torch.manual_seed(19)
        tensors=[torch.randn(1,n,8,dtype=torch.float64,requires_grad=True) for n in (256,384,384)]
        reference=[x.detach().clone().requires_grad_(True) for x in tensors]
        blocks=torch.tensor([[[[True,False,True],[False,True,False]],
                               [[False,True,True],[True,False,False]]]])
        actual=masked_block_sdpa(*tensors,2,blocks)
        q,k,v=[x.view(1,-1,2,4).transpose(1,2) for x in reference]
        mask=blocks.repeat_interleave(128,-2).repeat_interleave(128,-1)
        scores=(q@k.transpose(-1,-2))/2
        expected=(scores.masked_fill(~mask,-float('inf')).softmax(-1)@v).transpose(1,2).reshape(1,256,8)
        torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
        weight=torch.randn_like(actual)
        (actual*weight).sum().backward(); (expected*weight).sum().backward()
        for x,y in zip(tensors,reference): torch.testing.assert_close(x.grad,y.grad,rtol=1e-9,atol=1e-9)

if __name__=='__main__': unittest.main()
