"""Independent single-change DQN methods; baseline still uses original DQNAgent."""
from collections import deque, Counter
import numpy as np
import torch
from torch import nn
from .dqn import DQNAgent, ReplayBuffer


def legal_masks(states):
    # states B,C,H,W: no information beyond current wall map and position.
    b, _, h, w = states.shape
    flat = states[:, 1].flatten(1).argmax(1)
    row, col = flat // w, flat % w
    masks = []
    for dr, dc in ((-1,0),(0,1),(1,0),(0,-1)):
        rr, cc = row+dr, col+dc
        inside = (rr>=0)&(rr<h)&(cc>=0)&(cc<w)
        wall = states[torch.arange(b, device=states.device),0,rr.clamp(0,h-1),cc.clamp(0,w-1)] > .5
        masks.append(inside & ~wall)
    return torch.stack(masks,1)


def next_values(target_q, online_q=None, mask=None):
    selector = target_q if online_q is None else online_q
    if mask is not None:
        selector = selector.masked_fill(~mask, -torch.inf)
    actions = selector.argmax(1)
    values = target_q.gather(1, actions[:,None]).squeeze(1)
    if mask is not None:
        values = torch.where(mask.any(1), values, torch.zeros_like(values))
    return values


def method_targets(rewards, terminated, discounts, target_q, online_q=None, mask=None):
    bootstrap = next_values(target_q, online_q, mask)
    return rewards + discounts * torch.where(terminated, torch.zeros_like(bootstrap), bootstrap)


class NStep:
    def __init__(self, gamma, n=3):
        self.gamma, self.n, self.queue = gamma, n, deque()
        self.lengths = Counter()
        self.horizon_flush = 0
    def _pop(self):
        rows = list(self.queue)[:self.n]
        first, last = rows[0], rows[-1]
        k = len(rows)
        self.lengths[k] += 1
        reward = sum(self.gamma**j * row[2] for j,row in enumerate(rows))
        result = (first[0],first[1],reward,last[3],last[4],first[5],first[6],self.gamma**k)
        self.queue.popleft()
        return result
    def add(self, transition, episode_end):
        self.queue.append(transition)
        out = []
        if len(self.queue)>=self.n:
            out.append(self._pop())
        if episode_end:
            while self.queue:
                out.append(self._pop())
        return out
    def flush(self):
        out=[]
        while self.queue:
            out.append(self._pop())
        self.horizon_flush += len(out)
        return out


class MethodReplay(ReplayBuffer):
    def __init__(self, capacity,size,seed,method):
        super().__init__(capacity,size,seed)
        self.method = method
        self.general_capacity = 1800 if method=='success' else capacity
        self.general_count = self.general_position = self.success_count = self.success_position = 0
        self.priorities = np.ones(capacity,dtype=np.float64)
        self.discounts = np.ones(capacity,dtype=np.float32)
        self.success_labels = np.zeros(capacity,dtype=bool)
        self.episode = []
        self.total_insertions = self.success_insertions = self.sampled_success = self.sampled_reserved = 0
        self.sampled_duplicate_pairs = 0
        self.sampled_step_counts = Counter()
        self.completed_success_steps = set()
    def _write(self,i,row,success=False):
        state,action,reward,nxt,term,step,maze,discount=row
        self.states[i],self.actions[i],self.rewards[i]=state,action,reward
        self.next_states[i],self.terminated[i]=nxt,term
        self.insertion_steps[i],self.maze_seeds[i],self.discounts[i]=step,maze,discount
        self.success_labels[i]=success
        valid=self.valid_ids()
        self.priorities[i]=float(self.priorities[valid].max()) if len(valid) else 1.
    def valid_ids(self):
        return np.concatenate((np.arange(self.general_count),np.arange(1800,1800+self.success_count))) if self.method=='success' else np.arange(self.general_count)
    def insert(self,row):
        self._write(self.general_position,row)
        self.general_position=(self.general_position+1)%self.general_capacity
        self.general_count=min(self.general_count+1,self.general_capacity)
        self.count=self.general_count+self.success_count
        self.position=self.general_position
        self.total_insertions+=1
    def finish_success_episode(self,rows):
        steps={r[5] for r in rows}
        self.completed_success_steps.update(steps)
        self.success_labels[:self.general_count] |= np.isin(self.insertion_steps[:self.general_count],list(steps))
        for row in rows:
            self._write(1800+self.success_position,row,True)
            self.success_position=(self.success_position+1)%200
            self.success_count=min(self.success_count+1,200)
            self.success_insertions+=1
        self.count=self.general_count+self.success_count
    def distribution(self,beta):
        valid=self.valid_ids()
        p=self.priorities[valid]**.6
        return valid,p/p.sum()
    def sample_method(self,batch,step):
        valid=self.valid_ids()
        if self.method=='per':
            valid,p=self.distribution(.4+.6*min(step/20000,1))
            ids=self.rng.choice(valid,batch,replace=True,p=p)
            prob=p[ids]
            weights=(len(valid)*prob)**(-(.4+.6*min(step/20000,1)))
            weights/=weights.max()
        else:
            ids=self.rng.choice(valid,batch,replace=False)
            weights=np.ones(batch,dtype=np.float32)
        self.sampled_transitions+=batch
        self.sampled_step_counts.update(map(int,self.insertion_steps[ids]))
        self.sampled_goals+=int(self.terminated[ids].sum())
        self.sampled_success+=int(self.success_labels[ids].sum())
        self.sampled_reserved+=int((ids>=1800).sum()) if self.method=='success' else 0
        self.sampled_duplicate_pairs+=batch-len(np.unique(self.insertion_steps[ids]))
        data=[self.states[ids].astype(np.float32),self.actions[ids],self.rewards[ids],self.next_states[ids].astype(np.float32),self.terminated[ids],self.discounts[ids],weights.astype(np.float32)]
        return [torch.from_numpy(v) for v in data],ids
    def update_priorities(self,ids,errors):
        # Duplicate sampled slots: retain maximum absolute TD error in that batch.
        for i in np.unique(ids):
            self.priorities[i]=float(np.max(np.abs(errors[ids==i])))+1e-6
    def snapshot(self,current_step):
        valid=self.valid_ids()
        counts=Counter(map(int,self.maze_seeds[valid]))
        general_steps=set(map(int,self.insertion_steps[:self.general_count]))
        reserved_steps=set(map(int,self.insertion_steps[1800:1800+self.success_count])) if self.method=='success' else set()
        return {'count':self.count,'capacity':self.capacity,'general_occupancy':self.general_count,'success_occupancy':self.success_count,
                'duplicate_stored_transitions':len(general_steps&reserved_steps),'total_replay_insertions':self.total_insertions,
                'success_store_insertions':self.success_insertions,'known_success_episode_slots':int(self.success_labels[valid].sum()),
                'maze_counts':dict(counts),'sampling_totals':{'transitions':self.sampled_transitions,'goal_transitions':self.sampled_goals,
                'known_success_episode_transitions':self.sampled_success,'known_success_episode_fraction':self.sampled_success/self.sampled_transitions if self.sampled_transitions else None,
                'retrospective_success_episode_samples':sum(self.sampled_step_counts[s] for s in self.completed_success_steps),
                'retrospective_success_episode_fraction':sum(self.sampled_step_counts[s] for s in self.completed_success_steps)/self.sampled_transitions if self.sampled_transitions else None,
                'reserved_store_samples':self.sampled_reserved,'duplicate_transition_samples_in_batch':self.sampled_duplicate_pairs}}


class MethodAgent(DQNAgent):
    def __init__(self,config,seed):
        super().__init__(config,seed)
        self.method=config.train.method
        self.online.action_mask = self.method=='mask'
        self.replay=MethodReplay(config.train.replay_capacity,config.env.size,seed,self.method)
        self.nstep=NStep(config.train.gamma)
        self.pending_episode=[]
        self.step=0
    def observe(self,state,action,reward,nxt,terminated,truncated,step,maze):
        self.step=step
        row=(state.copy(),action,reward,nxt.copy(),terminated,step,maze,self.config.train.gamma)
        if self.method=='nstep':
            for item in self.nstep.add(row,terminated or truncated): self.replay.insert(item)
        else:
            self.replay.insert(row)
        if self.method=='success':
            self.pending_episode.append(row)
            if terminated:
                self.replay.finish_success_episode(self.pending_episode)
            if terminated or truncated:self.pending_episode=[]
    def flush(self):
        if self.method=='nstep':
            for row in self.nstep.flush():self.replay.insert(row)
    def act(self,state,step):
        if self.method!='mask':return super().act(state,step)
        if self.rng.random()<self.epsilon(step):
            mask=legal_masks(torch.from_numpy(state)[None])[0].numpy()
            return int(self.rng.choice(np.flatnonzero(mask)))
        from .dqn import greedy_action
        return greedy_action(self.online,state)
    def update(self):
        data,ids=self.replay.sample_method(self.config.train.batch_size,self.step)
        states,actions,rewards,nxt,terminated,discounts,weights=[v.to(self.device) for v in data]
        values=self.online(states).gather(1,actions[:,None]).squeeze(1)
        with torch.no_grad():
            targets=method_targets(rewards,terminated,discounts,self.target(nxt),self.online(nxt) if self.method=='double' else None,legal_masks(nxt) if self.method=='mask' else None)
        errors=targets-values
        loss=(nn.functional.smooth_l1_loss(values,targets,reduction='none')*weights).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(),self.config.train.gradient_clip,error_if_nonfinite=True)
        self.optimizer.step()
        if self.method=='per':self.replay.update_priorities(ids,errors.detach().cpu().numpy())
        return float(loss.detach())
