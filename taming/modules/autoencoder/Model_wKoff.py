import torch
import pytorch_lightning as pl
from transformers import AutoModel,AutoTokenizer
from taming.modules.baseline_related.A2BinderTokenizer import CommonTokenizer
import torch.nn as nn
import torch.nn.functional as F
from taming.modules.autoencoder.AgModule import AgMixPooler
from taming.modules.autoencoder.AbModule import AbEncoderLayer,AbPooler
from taming.modules.autoencoder.AbAgCross import CoAttentionBlock


class AbAgKerModel(pl.LightningModule): 
    def __init__(self,
                 H_llm="pretrain_model/Heavy_roformer",
                 L_llm="pretrain_model/Light_roformer",
                 ems2_llm="pretrain_model/esm2_t30",
                 ab_maxlen = 512, 
                 ag_maxlen = 640, 
                 dim = 512,
                 cdrs_layers = 4,
                 cdrs_heads = 16,
                 co_heads = 8,
                 ab_pool_nums = 128,
                 ag_pool_nums = 128,
                 ag_window = 9,
                 dp_out = 0.1,
                 num_experts = 8,
                 top_k = 2,
                 ):
        super().__init__()
        
        ### cite the mixed-model and loss params
        self.ab_maxlen = ab_maxlen  
        self.ag_maxlen = ag_maxlen

        # proteins and mols pretrained model   
        self.tokenizer = CommonTokenizer(logger=None, add_hyphen=False)
        self.heavy_tokenizer = self.tokenizer.get_bert_tokenizer(max_len=self.ab_maxlen, tokenizer_dir=H_llm)
        self.light_tokenizer = self.tokenizer.get_bert_tokenizer(max_len=self.ab_maxlen, tokenizer_dir=L_llm)
        self.HeavyModel = AutoModel.from_pretrained(H_llm, output_hidden_states=True, return_dict=True).eval()
        self.LightModel = AutoModel.from_pretrained(L_llm, output_hidden_states=True, return_dict=True).eval()

        self.antigen_tokenizer = AutoTokenizer.from_pretrained(ems2_llm,trust_remote_code=True,max_len = self.ag_maxlen)
        self.AntigenModel = AutoModel.from_pretrained(ems2_llm, output_hidden_states=True, return_dict=True).eval()
        
        self.feature_layer = Feature_Module_MOE(
            dim=dim,
            dp_out=dp_out,
            cdrs_layers=cdrs_layers,
            cdrs_heads=cdrs_heads,
            co_heads=co_heads,
            ab_pool_nums=ab_pool_nums,
            ag_pool_nums=ag_pool_nums,
            ag_window=ag_window,
            num_experts=num_experts,
            top_k=top_k,
        )
        self.koff_regressor = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.GELU(),
            nn.Dropout(dp_out),
            nn.Linear(dim*2, dim),
            nn.GELU(),
            nn.Dropout(dp_out),
            nn.Linear(dim, 1)
        )

    def forward(self, HLX, extra_HXL):  
        H_seq,L_seq,X_seq = HLX[0], HLX[1], HLX[2]
        ssf_x, cdrs_f = extra_HXL[0], extra_HXL[1]

        with torch.no_grad():
            H_info = self.heavy_tokenizer(H_seq, padding="max_length", max_length=self.ab_maxlen, truncation=True,add_special_tokens=False, return_tensors="pt").to(self.AntigenModel.device)
            L_info = self.light_tokenizer(L_seq, padding="max_length", max_length=self.ab_maxlen, truncation=True,add_special_tokens=False, return_tensors="pt").to(self.AntigenModel.device)
            H_outputs = self.HeavyModel(**H_info) 
            H_full_embeddings = H_outputs.last_hidden_state
            L_outputs = self.LightModel(**L_info) 
            L_full_embeddings = L_outputs.last_hidden_state
            h_att_mask, l_att_mask = H_info["attention_mask"], L_info["attention_mask"]

            X_info = self.antigen_tokenizer(X_seq, 
                                     max_length = self.ag_maxlen, 
                                     padding="max_length",
                                     truncation=True, 
                                     return_tensors="pt",
                                     add_special_tokens=False).to(self.AntigenModel.device)
                                     
            outputs = self.AntigenModel(**X_info) 
            Ag_emb = outputs.last_hidden_state
            ag_mask = X_info["attention_mask"]
        
        # antibody infomation concat
        ab_embs = torch.concat([H_full_embeddings, L_full_embeddings], dim=1)
        ab_mask = torch.concat([h_att_mask, l_att_mask],dim=1)

        output_kd, moe_output, aux_dict = self.feature_layer(Ag_emb, ab_embs, ag_mask, ab_mask, ssf_x, cdrs_f)
        output_koff = self.koff_regressor(moe_output)
        
        return output_kd.squeeze(dim=1), output_koff.squeeze(dim=1), aux_dict


class Namespace:
    def __init__(self, argvs):
        for k, v in argvs.items():
            setattr(self, k, v)
   

class Feature_Module_MOE(nn.Module):
    def __init__(self, 
                 dim,
                 dp_out, 
                 cdrs_heads,
                 cdrs_layers,
                 co_heads,
                 ab_pool_nums,
                 ag_pool_nums,
                 ag_window=10,
                 num_experts=8,
                 top_k=2,
                  ):
        super().__init__()
        

        self.ag_layer = nn.Sequential(
            nn.Linear(in_features=640, out_features=dim),
            nn.GELU(),
            nn.Dropout(dp_out),
            nn.LayerNorm(dim)
        )
        self.ab_layer = nn.Sequential(
            nn.Linear(in_features=768, out_features=dim),
            nn.GELU(),
            nn.Dropout(dp_out),
            nn.LayerNorm(dim)
        )
        
        self.Ag_pooler = AgMixPooler(target_len=ag_pool_nums, 
                                     window_T=ag_window, 
                                     pooling='topk') # ag_embs： B，token, 512 -> B，ag_pool_nums, 512
        self.cdrs_encoder_layer =nn.ModuleList([
            AbEncoderLayer(dim = dim, 
                            num_heads = cdrs_heads, 
                            ff_dim = 4 *dim, 
                            dropout = dp_out, 
                            attention_dropout = dp_out, 
                            activation_dropout = dp_out) 
                    for i in range(cdrs_layers)])

        self.ab_pooler = AbPooler(pooling_method='topk',
                                  topk=ab_pool_nums) # ab_embs： B，token, 512 -> B，ab_pool_nums, 512
        self.abag_cross_layer = CoAttentionBlock(dim = dim, 
                                                num_heads = co_heads, 
                                                ffn_dim = dim*4, 
                                                drop = dp_out, 
                                                attn_drop = dp_out, 
                                                act_drop = dp_out)

        self.dta_decoder = DTA_Decoder(dim = dim, 
                                       hidden_dim = dim, 
                                       num_experts = num_experts, 
                                       top_k = top_k, 
                                       dropout = dp_out) # dim: 512 -> 128

        
    def forward(self, ag_embs, ab_embs, ag_mask, ab_mask, ssf_x, cdrs_mask):

        # cross cls
        ag_embs = self.ag_layer(ag_embs)
        ab_embs = self.ab_layer(ab_embs)

        cdrs_mask = cdrs_mask.squeeze(1) # b,1,token -> b,token
        ssf_x = ssf_x.transpose(1,2) # b,7,token -> b,token,7
        ag_pooled, attn = self.Ag_pooler(ag_embs, ssf_x, ag_mask.bool())  # ag_embs： B，token, 512 -> B，ag_pool_nums, 512

        # ab ab_mask+cdrs based re-encoder
        for layer in self.cdrs_encoder_layer:
            ab_embs,attention_info = layer(ab_embs, mask=ab_mask, cdrs_score=cdrs_mask)
        ab_pooled = self.ab_pooler(ab_embs, attention_info) # ab_embs： B，token, 512 -> B，ab_pool_nums, 512
        
        x_b, x_g = self.abag_cross_layer(ab_pooled, ag_pooled, ag_all = ag_embs, pad_b = None, pad_g = ag_mask) # x_b/x_g : B，ab_pool_nums, 512 -> B，ab_pool_nums, 512

        x = torch.cat((x_b, x_g), dim=1) # b,T,E -> b,2T,E
        x = x.mean(dim=1) # b,2T,E -> b,E
        output_kd, moe_output,  aux = self.dta_decoder(x, return_aux=True) # b,E -> b,128
        return output_kd, moe_output, aux   


         
class DTA_Decoder(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts=4, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.norm_layer = nn.LayerNorm(dim)
        self.gate_network = nn.Linear(dim, num_experts)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim)
            ) for _ in range(num_experts)
        ])

        self.regressor = nn.Sequential(
            nn.Linear(dim, hidden_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, return_aux=False):
        x_pooled = self.norm_layer(x)
        gate_logits = self.gate_network(x_pooled)
        gate_probs = F.softmax(gate_logits, dim=1) # [B, E]

        expert_outputs = torch.stack([expert(x_pooled) for expert in self.experts], dim=1)
        moe_output = (gate_probs.unsqueeze(-1) * expert_outputs).sum(dim=1) 

        affinity_pred = self.regressor(moe_output)

        if return_aux:
            importance = gate_probs.sum(dim=0) / gate_probs.size(0)
            importance_loss = (self.num_experts * (importance ** 2)).sum()
            aux = {'importance_loss': importance_loss, 'gate_probs': gate_probs.detach()}
            return affinity_pred, moe_output, aux
        
        return affinity_pred, moe_output
