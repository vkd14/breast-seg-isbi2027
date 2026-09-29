import unittest
import numpy as np
import torch
from breastseg.core import (binary_mask, grayscale_uint8, Gabor, sampling_weights,
    validate_split, SegmentationLoss, logits_mask, metrics, boundary, Net, features)


class ScientificContracts(unittest.TestCase):
    def test_mask_encodings(self):
        m=np.array([[0,1],[1,0]],np.uint8)
        np.testing.assert_array_equal(binary_mask(m),binary_mask(m*255))
        np.testing.assert_allclose(sampling_weights([m,m*255],True),[1,1])

    def test_uniform_sampler_is_disabled(self):
        self.assertIsNone(sampling_weights([],False))

    def test_invalid_masks_fail(self):
        for m in (np.array([[np.nan]]),np.array([[-1]]),np.zeros((2,2,3))):
            with self.assertRaises(ValueError): binary_mask(m)

    def test_uint16_is_not_truncated(self):
        a=np.arange(64,dtype=np.uint16).reshape(8,8)
        self.assertGreater(grayscale_uint8(a).max(),200)
        self.assertEqual(grayscale_uint8(np.zeros((8,8))).max(),0)

    def test_gabor_finite_and_l1(self):
        for o,s,sigma in ((4,1,3),(8,3,5),(12,4,7)):
            g=Gabor(o,s,sigma)
            self.assertTrue(all(abs(np.abs(k).sum()-1)<1e-6 for k in g.kernels))
            self.assertTrue(np.isfinite(g(np.zeros((32,32),np.uint8))).all())
        self.assertEqual(len(Gabor(4,2,frequencies=[.1,.2]).kernels),8)
        with self.assertRaises(ValueError): Gabor(4,2,frequencies=[.1])

    def test_split_leakage_fails(self):
        a=dict(id='a',group='g',image_sha256='x',split='train')
        for key in ('id','group','image_sha256'):
            b=dict(id='b',group='h',image_sha256='y',split='test');b[key]=a[key]
            with self.assertRaises(ValueError): validate_split([a,b])
        validate_split([a,dict(id='b',group='h',image_sha256='y',split='test')])

    def test_logits_not_guessed_from_range(self):
        self.assertTrue(logits_mask(torch.tensor([.1])).item())
        self.assertFalse(logits_mask(torch.tensor([-.1])).item())
        self.assertFalse(logits_mask(torch.tensor([0.])).item())

    def test_loss_empty_full_and_extreme_gradients(self):
        for adaptive in (False,True):
            for target in (0.,1.):
                for value in (-100.,0.,100.):
                    z=torch.full((2,1,16,16),value,requires_grad=True)
                    loss=SegmentationLoss(.1,adaptive)(z,torch.full_like(z,target))
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(torch.isfinite(z.grad).all())

    def test_loss_nonfinite_fails(self):
        with self.assertRaises(FloatingPointError):
            SegmentationLoss()(torch.full((1,1,8,8),float('nan')),torch.zeros(1,1,8,8))

    def test_explicit_boundary_changes_loss(self):
        z=torch.zeros(1,1,8,8);t=torch.zeros_like(z);t[:,:,2:6,2:6]=1
        self.assertGreater(SegmentationLoss(.1)(z,t).item(),SegmentationLoss()(z,t).item())
        self.assertGreater(boundary(t).sum().item(),0)

    def test_native_metrics_and_empty_policy(self):
        z=np.zeros((16,16),np.uint8);a=z.copy();a[4:8,4:8]=1
        self.assertEqual(metrics(z,z)['dice'],1)
        self.assertEqual(metrics(z,a)['dice'],0)
        self.assertEqual(metrics(a,a)['boundary_f1_2px'],1)
        self.assertEqual(metrics(a,a)['hd95_px'],0)
        with self.assertRaises(ValueError): metrics(z,z[:2])

    def test_sobel_and_gabor_preserve_same_image_channels(self):
        image=np.arange(256,dtype=np.uint16).reshape(16,16)
        a=features(image,True,32,'gabor');b=features(image,True,32,'sobel')
        self.assertTrue(torch.equal(a[:3],b[:3]))
        self.assertTrue(torch.isfinite(a).all() and torch.isfinite(b).all())
        neutral=features(image,True,32,'neutral')
        self.assertTrue(torch.equal(a[:3],neutral[:3]))
        self.assertTrue(torch.equal(neutral[3],torch.zeros_like(neutral[3])))

    def test_residual_starts_as_exact_identity_on_image(self):
        model=Net('residual_gabor_b7',pretrained=False)
        model.base_model=torch.nn.Identity();model.eval()
        x=torch.randn(2,4,16,16)
        self.assertTrue(torch.equal(model(x),x[:,:3]))
        model(x).sum().backward()
        self.assertTrue(torch.isfinite(model.edge_gain.grad))

    def test_rle_decode_column_major(self):
        from breastseg.external import decode_rle
        np.testing.assert_array_equal(decode_rle('2 2',2,3),[[0,1,0],[1,0,0]])
        with self.assertRaises(ValueError): decode_rle('6 2',2,3)

if __name__=='__main__': unittest.main()
