"""Automated offline validation for the local federated SNN pipeline."""
import tempfile, unittest
import torch
from torch.utils.data import DataLoader
from dataset import SyntheticMedicalImages, split_dataset, make_federated_loaders
from model import SpikingCNN, parameter_count
from federated import fedavg, FederatedServer
from client import HospitalClient

class TestSpikeFed(unittest.TestCase):
    def test_dataset_and_preprocessing_shape(self):
        x,y=SyntheticMedicalImages(samples_per_class=2)[0]; self.assertEqual(tuple(x.shape),(1,28,28)); self.assertIsInstance(y,torch.Tensor)
    def test_lif_forward_and_surrogate_gradient(self):
        model=SpikingCNN(); x=torch.randn(2,1,28,28); spk,_=model(x,num_steps=2); self.assertEqual(tuple(spk.shape),(2,2,4)); spk.sum().backward(); self.assertTrue(any(p.grad is not None for p in model.parameters()))
    def test_parameter_budget(self): self.assertLess(parameter_count(SpikingCNN()),15000)
    def test_weighted_fedavg(self):
        result=fedavg([({"a":torch.tensor([1.])},1,{}),({"a":torch.tensor([5.])},3,{})]); self.assertAlmostEqual(result["a"].item(),4.)
    def test_round_and_checkpoint(self):
        ds=SyntheticMedicalImages(samples_per_class=3); train,_,_=split_dataset(ds); loaders=make_federated_loaders(train,2,batch_size=3)
        server=FederatedServer(); clients=[HospitalClient(i,l) for i,l in enumerate(loaders)]; row=server.run_round(clients)
        self.assertEqual(len(row["clients"]),2); self.assertGreater(row["communication_bytes"],0)
        path=server.save("test_results.json"); self.assertTrue(torch.load("checkpoints/global_model.pt", weights_only=True))
    def test_client_update_has_no_raw_images(self):
        loader=DataLoader(SyntheticMedicalImages(samples_per_class=2),batch_size=2)
        client=HospitalClient("privacy-check",loader); state,n,report=client.fit(client.model.state_dict())
        self.assertFalse(any(tuple(value.shape)==(1,28,28) for value in state.values()))
        self.assertNotIn("images",report); self.assertNotIn("labels",report)
if __name__ == "__main__": unittest.main(verbosity=2)
