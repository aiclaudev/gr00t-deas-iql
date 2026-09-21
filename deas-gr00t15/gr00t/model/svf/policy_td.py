"""SVF with a learned double Q and policy-sampled chunk TD backup (no IQL V)."""
from copy import deepcopy
import torch
from torch import nn
from .adapters import prepare_head_context, velocity_from_head
from .lora import ActorTuningConfig, apply_dit_lora
from .networks import DoubleSoftValue
from .objective import SVFConfig, joint_losses
from gr00t.model.iql.core import masked_mean, td_target


def policy_q_loss(q1, q2, returns, bootstrap_mask, next_q1, next_q2, valid, discount=.99, horizon=16):
    target = td_target(returns, bootstrap_mask, torch.minimum(next_q1, next_q2), discount, horizon).detach()
    loss = masked_mean(((q1-target).square() + (q2-target).square()) / 2, valid)
    return loss, target


class PolicyTDSVF(nn.Module):
    def __init__(self, actor_path, config=SVFConfig(), rank=16, discount=.99, tau=.005):
        super().__init__()
        from gr00t.model.gr00t_n1 import GR00T_N1_5
        from gr00t.model.action_head.deas_critic import CategorySpecificMLP
        from gr00t.model.critic.networks import DoubleCritic
        self.svf_config=config; self.discount=discount; self.tau=tau
        self.actor=GR00T_N1_5.from_pretrained(str(actor_path), tune_llm=False, tune_visual=False,
            tune_projector=True, tune_diffusion_model=True, torch_dtype=torch.float32, local_files_only=True)
        if (self.actor.action_horizon,self.actor.action_dim)!=(16,32):
            raise ValueError('Expected RoboCasa GR00T 1.5 chunk16, padded action32')
        self.actor.backbone.requires_grad_(False).to(torch.bfloat16).eval()
        self.actor.action_head.float()
        self.reference_head=deepcopy(self.actor.action_head).requires_grad_(False).to(torch.bfloat16).eval()
        self.actor_tuning=ActorTuningConfig('dit-lora',rank, float(rank),0.)
        self.lora_target_modules=apply_dit_lora(self.actor.action_head,self.actor_tuning)
        width=self.actor.action_head.config.backbone_embedding_dim
        self.projection=CategorySpecificMLP(32,width,1024,64)
        self.q=DoubleCritic(64+64+16*32,[512]*4,output_dim=1)
        self.target_projection=deepcopy(self.projection).requires_grad_(False)
        self.target_q=deepcopy(self.q).requires_grad_(False)
        self.soft_value=DoubleSoftValue(feature_dim=64,state_dim=64,action_horizon=16,action_dim=32,
                                        time_dim=64,hidden_dim=512,depth=4,loss_type='mse')
        self.reference_microbatch_size=16
        self.train()

    def train(self,mode=True):
        super().train(mode)
        if hasattr(self,'actor'):
            self.actor.backbone.eval();self.actor.action_head.eval()
            self.actor.action_head.model.train(mode)
        for name in ['reference_head','target_q','target_projection']:
            if hasattr(self,name):getattr(self,name).eval()
        return self

    @torch.no_grad()
    def update_targets(self):
        for source,target in [(self.q,self.target_q),(self.projection,self.target_projection)]:
            for p,tp in zip(source.parameters(),target.parameters()):tp.lerp_(p,self.tau)

    def project(self,pooled,ids,target=False):
        layer=self.target_projection if target else self.projection
        with torch.autocast('cuda',dtype=torch.bfloat16):out=layer(pooled,ids)
        return out.float().tanh()

    @torch.no_grad()
    def sample_next(self,raw,inputs):
        head=self.actor.action_head
        prior=head.training;head.eval()
        try:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                context=prepare_head_context(head,raw,inputs)
                x=torch.randn((inputs['state'].shape[0],16,32),device=inputs['state'].device)
                for i in range(self.svf_config.flow_steps):
                    x=x+velocity_from_head(head,context,x,i/self.svf_config.flow_steps).float()/self.svf_config.flow_steps
            return x.clamp(-1,1)
        finally:
            head.train(prior)
            # Only the DiT trains; the frozen preprocessing remains in eval mode.
            if self.training:
                head.eval();head.model.train()

    def forward(self,batch):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            raw=self.actor.backbone(batch)
            nxt=self.actor.backbone(batch,eagle_prefix='next_eagle_')
            reference=prepare_head_context(self.reference_head,raw,batch)
            next_inputs=dict(batch,state=batch['next_state'])
            next_reference=prepare_head_context(self.reference_head,nxt,next_inputs)
            pooled=reference.backbone_features.mean(1,keepdim=True).float()
            next_pooled=next_reference.backbone_features.mean(1,keepdim=True).float()
            next_actions=self.sample_next(nxt,next_inputs)*batch['action_mask'].float()
        states=batch['state'].float()*batch['state_mask'].float()
        next_states=batch['next_state'].float()*batch['next_state_mask'].float()
        actions=batch['action'].float()*batch['action_mask'].float()
        features=self.project(pooled,batch['embodiment_id'])
        with torch.no_grad():
            target_features=self.project(pooled,batch['embodiment_id'],True)
            next_features=self.project(next_pooled,batch['embodiment_id'],True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                tq1,tq2=self.target_q(next_features,next_states,next_actions)
        with torch.autocast('cuda',dtype=torch.bfloat16):q1,q2=self.q(features,states,actions)
        qloss,target=policy_q_loss(q1.float(),q2.float(),batch['chunk_return'].float(),
            batch['bootstrap_mask'],tq1.float(),tq2.float(),batch['chunk_valid'],self.discount,16)
        actor_context=None
        def actor_velocity(x,t):
            nonlocal actor_context
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if actor_context is None:actor_context=prepare_head_context(self.actor.action_head,raw,batch)
                return velocity_from_head(self.actor.action_head,actor_context,x,t).float()
        def reference_velocity(x,t):
            context=reference.repeat_batches(x.shape[0]//actions.shape[0])
            result=[]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for start in range(0,len(x),self.reference_microbatch_size):
                    end=start+self.reference_microbatch_size
                    result.append(velocity_from_head(self.reference_head,context.slice_batch(start,end),x[start:end],t[start:end]).float())
            return torch.cat(result)
        def teacher_score(endpoints):
            k,b,h,d=endpoints.shape
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                a=endpoints.reshape(k*b,h,d)*batch['action_mask'].repeat(k,1,1)
                v1,v2=self.target_q(target_features.repeat(k,1,1),states.repeat(k,1,1),a)
                return torch.minimum(v1,v2).float().reshape(k,b)
        svloss,metrics=joint_losses(self.soft_value,actor_velocity,reference_velocity,teacher_score,
            target_features,states,actions,batch['action_mask'],self.svf_config)
        metrics.update(q_loss=qloss.detach(),td_abs_error=(q1.float().detach()-target).abs().mean(),
                       terminal_fraction=(1-batch['bootstrap_mask']).float().mean())
        for name,x in [('q1',q1),('q2',q2),('td_target',target)]:
            x=x.detach().float();metrics.update({name+'_min':x.min(),name+'_mean':x.mean(),name+'_max':x.max()})
        return {'loss':qloss+svloss,'metrics':metrics}
