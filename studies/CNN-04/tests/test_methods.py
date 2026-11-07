import unittest
import numpy as np
import torch
from rl_lab.methods import NStep,MethodReplay,method_targets,legal_masks,next_values
from rl_lab.config import from_dict
from rl_lab.dqn import DQNAgent,seed_everything


def row(step=1,reward=0.,term=False):
    s=np.zeros((3,3,3),dtype=np.float32);s[1,0,0]=1
    return (s,1,reward,s.copy(),term,step,1,.99)

class MethodsTests(unittest.TestCase):
    def test_per_probabilities_weights_and_updates(self):
        b=MethodReplay(4,3,3,'per')
        for j in range(4):b.insert(row(j+1))
        b.priorities[:]=[1,2,4,8]
        ids,p=b.distribution(.4)
        np.testing.assert_allclose(p,np.array([1,2,4,8])**.6/sum(np.array([1,2,4,8])**.6))
        data,ids=b.sample_method(20,10000)
        expected=(4*p[ids])**(-.7);expected/=expected.max()
        np.testing.assert_allclose(data[-1].numpy(),expected,rtol=1e-6)
        b.update_priorities(np.array([1,1,2]),np.array([-2.,5.,-3.]))
        self.assertAlmostEqual(b.priorities[1],5.000001)
        maximum=b.priorities.max(); b.insert(row(5));self.assertEqual(b.priorities[0],maximum)
        empty=MethodReplay(4,3,0,'per');empty.insert(row());self.assertEqual(empty.priorities[0],1)
        for step,beta in ((0,.4),(20000,1.)):
            data,ids=b.sample_method(20,step)
            _,p=b.distribution(beta)
            expected=(4*p[ids])**(-beta);expected/=expected.max()
            np.testing.assert_allclose(data[-1].numpy(),expected,rtol=1e-6)
    def test_success_fifo_slots_uniform_duplicate(self):
        b=MethodReplay(2000,3,0,'success')
        rows=[row(i) for i in range(1,101)]
        for r in rows:b.insert(r)
        b.finish_success_episode(rows)
        self.assertEqual(b.count,200);self.assertEqual(b.snapshot(100)['duplicate_stored_transitions'],100)
        data,ids=b.sample_method(200,100)
        self.assertEqual(sum(ids>=1800),100)
        self.assertEqual(b.sampled_success,200)
        self.assertEqual(b.sampled_duplicate_pairs,100)
        for i in range(101,2001):b.insert(row(i))
        self.assertEqual(b.general_count,1800);self.assertEqual(b.success_count,100)
        self.assertEqual(b.snapshot(2000)['duplicate_stored_transitions'],0)
        b.finish_success_episode([row(i) for i in range(2001,2301)])
        self.assertEqual(b.success_count,200)
        self.assertEqual(set(b.insertion_steps[1800:2000]),set(range(2101,2301)))
    def test_nstep_terminal_truncation_horizon(self):
        n=NStep(.5)
        self.assertEqual(n.add(row(1,1),False),[])
        self.assertEqual(n.add(row(2,2),False),[])
        out=n.add(row(3,4,True),True)
        self.assertEqual([r[2] for r in out],[3.,4.,4.])
        self.assertEqual([r[7] for r in out],[.125,.25,.5])
        self.assertTrue(all(r[4] for r in out))
        self.assertEqual(len(n.queue),0)
        n.add(row(4,2),False)
        out=n.add(row(5,6),True) # truncated, not terminated
        self.assertEqual(out[0][2],5);self.assertFalse(out[0][4])
        self.assertEqual(out[0][7],.25)
        targets=method_targets(torch.tensor([out[0][2]]),torch.tensor([False]),torch.tensor([out[0][7]]),torch.tensor([[8.,1.,2.,3.]]))
        self.assertEqual(targets.item(),7.)
        targets=method_targets(torch.tensor([5.]),torch.tensor([True]),torch.tensor([.25]),torch.tensor([[8.,1.,2.,3.]]))
        self.assertEqual(targets.item(),5.)
        n.add(row(6,9),False);out=n.flush()
        self.assertEqual(out[0][7],.5);self.assertEqual(n.horizon_flush,1)
    def test_double_selection_evaluation_and_terminal_mask(self):
        tq=torch.tensor([[1.,10.,2.,3.],[9.,8.,7.,6.]])
        oq=torch.tensor([[20.,0.,1.,2.],[1.,2.,30.,4.]])
        result=method_targets(torch.tensor([1.,1.]),torch.tensor([False,True]),torch.tensor([.5,.5]),tq,oq)
        torch.testing.assert_close(result,torch.tensor([1.5,1.]))
        mask=torch.tensor([[False,False,True,True],[False,False,False,False]])
        result=method_targets(torch.tensor([1.,1.]),torch.tensor([False,True]),torch.ones(2),tq,None,mask)
        torch.testing.assert_close(result,torch.tensor([4.,1.]))
        self.assertTrue(torch.isfinite(result).all())
    def test_mask_walls_boundaries_and_actions(self):
        s=row()[0];s[0,0,1]=1
        mask=legal_masks(torch.from_numpy(s)[None])[0]
        self.assertEqual(mask.tolist(),[False,False,True,False])
        from rl_lab.methods import MethodAgent
        raw={'env':{'size':3},'train':{'method':'mask','steps':100,'learning_starts':64}}
        seed_everything(0);a=MethodAgent(from_dict(raw),0)
        for i in range(30):self.assertEqual(a.act(s,i),2)
        a.online.action_mask=True
        from rl_lab.dqn import greedy_action
        self.assertEqual(greedy_action(a.online,s),2)
    def test_baseline_reproducibility(self):
        config=from_dict({'env':{'size':3},'train':{'steps':100}})
        seed_everything(0);a=DQNAgent(config,0)
        for i in range(64):a.replay.add(*row(i)[:5],i,1)
        seed_everything(0);b=DQNAgent(config,0)
        for i in range(64):b.replay.add(*row(i)[:5],i,1)
        self.assertEqual(a.update(),b.update())
        for pa,pb in zip(a.online.parameters(),b.online.parameters()):torch.testing.assert_close(pa,pb,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
