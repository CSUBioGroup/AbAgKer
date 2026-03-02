import torch
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F
from main_wandb import instantiate_from_config
from taming.modules.metrics.metrics import CI, RM2

class AbAgKerTrainer(pl.LightningModule): 
    def __init__(self,
                 model_config,
                 opt_config,
                 loss_config,
                 monitor="kd_rmse",
                 train_type="kd",
                 ### scheduler config
                 learning_rate=None,
                 ckpt_path=None,
                 ignore_keys=[]
                 ):
        super().__init__()

        self.train_type = train_type
        self.opt_config = opt_config
        self.monitor = monitor
        self.ag_maxlen = model_config.get('ag_maxlen', 896) 
        self.ab_maxlen = model_config.get('ab_maxlen', 256) 

        # proteins and mols pretrained model   
        self.loss = instantiate_from_config(loss_config)    
        self.model = instantiate_from_config(model_config)    

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.learning_rate = learning_rate

        self.kd_metrics = torchmetrics.MetricCollection({
            "kd_pearson": torchmetrics.PearsonCorrCoef(),
            "kd_spearman": torchmetrics.SpearmanCorrCoef(),
            "kd_rm2": RM2(),
            "kd_ci": CI(),
            "kd_mse": torchmetrics.MeanSquaredError(),
            "kd_rmse": torchmetrics.MeanSquaredError(squared=False),
        }, prefix="val/")
        
        self.temp_metrics = {}
        self.best_monitor_value = float('inf')
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):

        HLX, ex_HXL, label = self.get_input(batch)
        kd, koff, AbAgI = label[0], label[1], label[2]
        kd_pre, _, aux_dict = self.model(HLX,ex_HXL)

        if optimizer_idx == 0: 
            total_loss, log_dict = self.loss(self.train_type, kd_pre, None, kd, koff, aux_dict, optimizer_idx, split="train") 
            self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            if not torch.isfinite(total_loss.detach()):
                print(f"[NaN] rank={self.global_rank}, step={self.global_step} -> use zero loss (connected)")
                total_loss = sum((p.sum() for p in self.parameters() if p.requires_grad)) * 0.0
                return total_loss  

            return total_loss

    def validation_step(self, batch, batch_idx):
        
        HLX, extra_HXL, label = self.get_input(batch)
        kd, koff, AbAgI = label[0], label[1], label[2]
        kd_pre, _, _ = self.model(HLX,extra_HXL)
        self.kd_metrics.update(kd_pre, kd)

    def validation_epoch_end(self,outputs):
        metrics = self.kd_metrics.compute()
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        current_monitor_value = metrics.get(self.monitor, None)
        monitor_value = current_monitor_value.item()
        if monitor_value < self.best_monitor_value:
            print(f"****New best monitor value: {monitor_value} at epoch {self.current_epoch}")
            self.best_monitor_value = monitor_value
            self.temp_metrics = {
                'epoch': self.current_epoch,
                'monitor_name': self.monitor,
                'monitor_value': monitor_value,
                'metrics': {}
            }
            for metric_name, metric_value in metrics.items():
                self.temp_metrics['metrics'][metric_name] = metric_value.item()
    
        self.kd_metrics.reset()

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        grad_clip_val = self.opt_config.get('grad_clip_val', 1.0)
        torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_val)

    def get_input(self, batch):

        pdb_name = batch["pdb"]
        kd = batch["AbAgA"].to(torch.float32)
        kon = batch["AbAgI"].to(torch.float32)  
        koff = batch["AbAgAoff"].to(torch.float32)
        AbAgI = batch["AbAgI"].to(torch.float32)

        Ab_H = [" ".join(self.model.tokenizer.split(i)) for i in batch["H"]]
        Ab_L = [" ".join(self.model.tokenizer.split(i)) for i in batch["L"]]
        Ag_X = batch["X"]

        Ag_X_e = []
        cdrs_f = []
        for i in range(len(pdb_name)):
            seq_f = torch.load("/root/private_data/luog/Data_AbAg/AbAgKer_all/ssf/"+pdb_name[i]+".pt")
            cdr_f = torch.load("/root/private_data/luog/Data_AbAg/AbAgKer_all/cdrs_ssf/cdrs_5/"+pdb_name[i]+".pt")
            if cdr_f.sum() == 0: # is cdr is error, usingh ssf cover for simulate cdr mask
                cdr_f = torch.zeros(512)
                if seq_f["H"] is not None:
                    cdr_f[:seq_f["H"].shape[1]] = seq_f["H"][2,:]
                if seq_f["L"] is not None:
                    cdr_f[256:256+seq_f["L"].shape[1]] = seq_f["L"][2,:]
                cdr_f = (cdr_f>0.9).long()
                cdr_f = cdr_f.unsqueeze(0)
            
            X_ssf = seq_f["X"]
            X_ssf = F.pad(X_ssf, (0, self.ag_maxlen - X_ssf.shape[1]), "constant", 0) if X_ssf.shape[1] < self.ag_maxlen else X_ssf[:,:self.ag_maxlen]
            Ag_X_e.append(X_ssf)
            cdrs_f.append(cdr_f)

        ssf_x = torch.stack(Ag_X_e).to(self.device)
        cdrs_f = torch.stack(cdrs_f).to(self.device)
        return (Ab_H, Ab_L, Ag_X), (ssf_x, cdrs_f), (kd, koff, AbAgI)

    def configure_optimizers(self):
        for param in self.model.AntigenModel.parameters():
            param.requires_grad = False
        for param in self.model.HeavyModel.parameters():
            param.requires_grad = False
        for param in self.model.LightModel.parameters():
            param.requires_grad = False
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        betas = (self.opt_config.get('beta1', 0.9), self.opt_config.get('beta2', 0.999))
        optimizer_groups = [
            {'params': trainable_params, 'lr': self.learning_rate}
        ]
        opt_mixed = torch.optim.Adam(optimizer_groups, betas=betas)


        from taming.modules.lr_scheduler import LambdaWarmUpCosineScheduler
        if self.opt_config.get('warm_up_steps') is not None:
            scheduler = LambdaWarmUpCosineScheduler(
                warm_up_steps=self.opt_config.get('warm_up_steps', 50),
                lr_min=self.opt_config.get('lr_min', 1e-6),
                lr_max=self.opt_config.get('lr_max', 1e-4),
                lr_start=self.opt_config.get('lr_start', 1e-5),
                max_decay_steps=self.opt_config.get('max_decay_steps', 1200),
                verbosity_interval=self.opt_config.get('verbosity_interval', 100)
            )

            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(opt_mixed, lr_lambda=scheduler),
                'interval': 'step',
                'frequency': 10, # every 10-step update the lr
            }
            return [opt_mixed], [lr_scheduler]
        else:
            return [opt_mixed], [] 
    

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}") 

