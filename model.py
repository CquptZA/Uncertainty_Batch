import torch
import torch.nn as nn
import pdb
from layers import *
from util import *

##CLIF
class CLIFModel(nn.Module):
    def __init__(self,configs):
        super(CLIFModel,self).__init__()
        self.rand_seed=configs['seed']

        #定义label_emb
        self.label_emb=nn.Parameter(torch.eye(configs['num_classes']),requires_grad=False)
        #定义label_edge
        self.label_edge=nn.Parameter(torch.eye(configs['num_classes']),requires_grad=False)

        self.hidden=768
        self.creation1=nn.MultiLabelSoftMarginLoss()
        self.creation2=LinkPredictionLoss_cosine()

        self.GIN_encoder=GIN(configs['num_layers'],input_size=configs['num_classes'],out_size=configs['class_emb_size'],
                                hidden_list=configs['hidden_list'])

        self.FD_model=FDModel(input_x_size=configs['input_x_size'],input_y_size=configs['class_emb_size'],hidden_size=self.hidden,out_size=self.hidden,
                                in_layers1=configs['in_layers'],out_layers=1,nonlinearity='leaky_relu',drop_ratio=0.2,negative_slope=0.1,batchNorm=False)
                                
        self.cls_conv=nn.Conv1d(configs['num_classes'],configs['num_classes'],self.hidden,groups=configs['num_classes'])
        self.reset_parameters()
        
    def reset_parameters(self):
        Init_random_seed(self.rand_seed)
        nn.init.normal_(self.label_emb)
        self.GIN_encoder.reset_parameters()
        self.FD_model.reset_parameters()
        self.cls_conv.reset_parameters()
        
    def get_config_optim(self):
        return [{'params': self.GIN_encoder.parameters()},
                {'params': self.FD_model.parameters()},
                {'params': self.cls_conv.parameters()}]
    def forward(self,x):
        label_emb=self.GIN_encoder(self.label_emb,self.label_edge)

        x=self.FD_model(x,label_emb)

        out=self.cls_conv(x).squeeze(dim=2)
        return out,label_emb

    def loss_function_train(self,outputs,label):
        loss=self._compute_loss(*outputs,label)
        return {"Loss":loss}

    def _compute_loss(self,output,emb,label):
        adj=self.label_edge.data + torch.eye(self.label_edge.data.size(0),
                                                        dtype=self.label_edge.data.dtype,
                                                        device=self.label_edge.data.device)
        loss = self.creation1(output,label)+1e-3*self.creation2(emb,adj)
        return loss
    def _loss_per_label(self, preds, targets):
        individual_losses = F.multilabel_soft_margin_loss(preds, targets, reduction='none')
        return individual_losses
    
    def custom_multilabel_soft_margin_loss(self,preds, targets, weights):
        individual_losses = F.multilabel_soft_margin_loss(preds, targets, reduction='none')
        weights_tensor = torch.tensor(weights, dtype=individual_losses.dtype, device=individual_losses.device)
        weighted_losses = individual_losses * weights_tensor
        return weighted_losses

    
    def predict(self,x):
        self.eval()
        with torch.no_grad():
            label_emb=self.GIN_encoder(self.label_emb,self.label_edge)

            x=self.FD_model(x,label_emb)

            out=self.cls_conv(x).squeeze(dim=2).sigmoid_()
        return label_emb,out



##DELA
class DELA(nn.Module):
    def __init__(self, configs):
        super(DELA, self).__init__()
        self.configs = configs
        init_random_seed(self.configs['seed'])
        
        # Embedding function
        self.encoder = MLP(configs['in_features'], 256, [256, 512], False,
                           configs['drop_ratio'], "relu")
        self.fc_mu = nn.Linear(256, configs['latent_dim'])
        
        # Standard deviation function to parametrize the noise distribution 
        # (share the first three layers with the embedding function)
        self.fc_logvar = nn.Linear(256, configs['latent_dim'])
        
        # Function to parametrize the binary Concrete gates
        self.logit = nn.Parameter(torch.randn(configs['num_classes'], configs['latent_dim']))
        self.scale_layer = nn.Linear(configs['latent_dim'], configs['latent_dim'])
        
        # Classifiers
        self.decoder = MLP(configs['in_features']+configs['latent_dim'], 512,
                           [256], False, nonlinearity="relu")
        self.classifier = nn.Conv1d(configs['num_classes'], configs['num_classes'], 512,
                                    groups=configs['num_classes'])
        
        # Move model to the right device for consistent initialization
        self.to(configs['device'])
        self.training=True
        
        self.reset_parameters()

        
    def reset_parameters(self):
        init_random_seed(self.configs['seed'])
        self.encoder.reset_parameters()
        self.fc_mu.reset_parameters()
        self.fc_logvar.reset_parameters()
        self.decoder.reset_parameters()
        self.classifier.reset_parameters()
        self.logit.data.uniform_(-10, 10)
        self.scale_layer.reset_parameters()
        nn.init.constant_(self.scale_layer.bias, 2.0)
    
    def get_config_optim(self):
        return [{'params': self.encoder.parameters()},
                {'params': self.fc_mu.parameters()},
                {'params': self.fc_logvar.parameters()},
                {'params': self.decoder.parameters()},
                {'params': self.classifier.parameters()},
                {'params': self.logit, 'lr': self.configs['lr_ratio']*self.configs['lr']},
                {'params': self.scale_layer.parameters(), 'lr': self.configs['lr_ratio']*self.configs['lr']}]
    
    def forward(self, input: Tensor) -> Tuple[Tensor, ...]:
        # Obtain latent representation of data and standard deviation of the noise distribution [B x D]
        z, n_logvar = self._encode(input)#[B,D]
        
        # Sample the indicator vector of non-informative features for each class label from binary Concrete gates
        if self.training:
            logit = self.scale_layer(self.logit) # [Q x D]
            samples = gumbel_sigmoid(logit, tau=self.configs['tau'], gumbel_noise=True, hard=True) # [Q x D]
             # For numerical stability when calculating the KL-divergence and smoother decision boundary
            #samples = samples.clamp(min=self.configs['off_noise']).detach() + samples - samples.detach()
        else:
            samples = None
            
        # Perturb latent representation
        z_k = self._add_noise(z, n_logvar, samples) # [B x Q x D]
        
        # Classification
        preds = self._decode(z_k, input) # [B x Q]
        
        return z, n_logvar, samples, preds
    
    
    def loss_function_train(self, preds: Tuple[Tensor, ...], targets: Tensor) -> dict:
        Loss, Kl_loss, Cls_loss = self._compute_loss(*preds, targets) 
        
        return {'Loss': Loss,
                'Kl_loss': Kl_loss,
                'Cls_loss': Cls_loss}
    
    def loss_function_eval(self, preds: Tuple[Tensor, ...], targets: Tensor) -> dict:
        Loss, _, Cls_loss = self._compute_loss(*preds, targets)
        
        return {'Loss': Loss.detach().item(),
                'Cls_loss': Cls_loss.detach().item()}
    
    def predict(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        self.eval()
        with torch.no_grad():
            # Obtain latent representation of data [B x D]
            x_mu, _ = self._encode(input)
            z_x = self._add_noise(x_mu, None, None) # [B x Q x D]
            
            # Classification
            pred_probs = self._decode(z_x, input).sigmoid_() # [B x Q]
            pred_labels = (pred_probs > 0.5).type_as(pred_probs) # [B x Q]
        
        return pred_labels, pred_probs
        
    def _encode(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        result = self.encoder(input)
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)
        
        return mu, logvar

    def _add_noise(self, z: Tensor, n_logvar: Tensor, samples: Tensor=None):
        if samples is not None:
            std = torch.exp(0.5 * n_logvar) # sigma = exp(0.5 * log(sigma^2))
            eps = torch.randn_like(std)
            z_k = z.unsqueeze(1) + samples.unsqueeze(0) * std.unsqueeze(1) * eps.unsqueeze(1) # [B x Q x D]
        else:
            z_k = z.unsqueeze(1).expand(-1, self.configs['num_classes'], -1) # [B x Q x D]
            
        return z_k
    
    def _decode(self, z: Tensor, input: Tensor) -> Tensor:
        # Original feature is incorporated for more stable training. Similar technique has been used in Conditional VAE and MPVAE
        z = self.decoder(torch.cat([input.unsqueeze(1).expand(-1, self.configs['num_classes'], -1), z],
                                   dim=2)) # [B x Q x D]
        preds = self.classifier(z).squeeze(2) # [B x Q]
        
        return preds
    
    def _compute_loss(self, z: Tensor, n_logvar: Tensor, samples: Tensor,
                      preds: Tensor, targets: Tensor) -> Tuple[Tensor, ...]:
        # Classification loss
        Cls_loss = F.multilabel_soft_margin_loss(preds, targets) * targets.size(1)

        
        # KL-divergence loss (Constraint on noise distribution)
        if samples is not None:
            Kl_loss = self._KL(z, n_logvar, samples)
            Loss = Cls_loss + self.configs['beta'] * Kl_loss

        else:
            Kl_loss = None
            Loss = Cls_loss
            
        return Loss, Kl_loss, Cls_loss
    
    def _loss_per_label(self, preds, targets):
        individual_losses = F.multilabel_soft_margin_loss(preds, targets, reduction='none')
        return individual_losses
    def _loss_per_label2(self, z: Tensor, n_logvar: Tensor, samples: Tensor,
                      preds: Tensor, targets: Tensor) -> Tuple[Tensor, ...]:

        Cls_loss = torch.mean(F.multilabel_soft_margin_loss(preds, targets, reduction='none'), dim=1)

        if samples is not None:
            Kl_loss = self._KL1(z, n_logvar, samples)
            Loss = Cls_loss + self.configs['beta'] * Kl_loss

        else:
            Kl_loss = None
            Loss = Cls_losss
        return Loss

    def custom_multilabel_soft_margin_loss(self,preds, targets, weights):
#         individual_losses = F.multilabel_soft_margin_loss(preds, targets, reduction='none')
#         return individual_losses
        sigmoid_preds = torch.sigmoid(preds)
        weights_tensor = torch.tensor(weights, dtype=preds.dtype, device=preds.device)
        losses = - (targets * torch.log(sigmoid_preds + 1e-6) + (1 - targets) * torch.log(1 - sigmoid_preds + 1e-6))
        loss_per_sample =weights_tensor * losses.sum(dim=1)
        return loss_per_sample
         
    def _KL(self, z: Tensor, n_logvar: Tensor, samples: Tensor):
        z = z.unsqueeze(1)
        n_logvar = n_logvar.unsqueeze(1)
        samples = samples.unsqueeze(0)
        KL_mat = -n_logvar - 2*torch.log(samples+1e-6) - 1 + torch.exp(n_logvar)*samples**2 + z**2 # [B x Q x D]
        return torch.mean(0.5*torch.sum(KL_mat, dim=2))

    def _KL1(self, z: Tensor, n_logvar: Tensor, samples: Tensor):
        z = z.unsqueeze(1)
        n_logvar = n_logvar.unsqueeze(1)
        samples = samples.unsqueeze(0)
        KL_mat = -n_logvar - 2*torch.log(samples+1e-6) - 1 + torch.exp(n_logvar)*samples**2 + z**2 # [B x Q x D]
#         return torch.mean(0.5*torch.sum(KL_mat, dim=2))
        KL_mean_per_sample = torch.mean(KL_sum, dim=1) 
        return KL_mean_per_sample
    
    

class PACA(nn.Module):
    def __init__(self, configs):
        super(PACA, self).__init__()
        self.configs = configs
        init_random_seed(self.configs['rand_seed'])
        
        # Probabilistic autoencoder for features
        self.encoder = MLP(configs['in_features'], 256, [256, 512], False,
                           configs['drop_ratio'], "relu")
        self.fc_mu = nn.Linear(256, configs['latent_dim'])
        self.fc_logvar = nn.Linear(256, configs['latent_dim'])
        self.decoder = MLP(configs['latent_dim'], configs['in_features'],
                           [256, 512, 256], False, nonlinearity="relu",
                           with_output_nonlineartity=False)
        
        # Probabilistic autoencoder for labels
        self.label_encoder = MLP(configs['num_classes'], 256, [512], False, 
                                 configs['drop_ratio'], "relu")
        self.label_fc_mu = nn.Linear(256, configs['latent_dim'])
        self.label_fc_logvar = nn.Linear(256, configs['latent_dim'])
        self.label_decoder = MLP(configs['latent_dim'], 512, [256], False,
                                 nonlinearity="relu")
        self.label_classifier = nn.Linear(512, configs['num_classes'])
        
        # Probabilistic prototypes via normalizing flows
        self.label_encodings = nn.Parameter(torch.eye(configs['num_classes']).unsqueeze(0),
                                            requires_grad=False)
        base_dist = torch.distributions.normal.Normal(torch.zeros(configs['latent_dim']).to(configs['device']),
                                                      torch.ones(configs['latent_dim']).to(configs['device']))
        self.pos_prototypes = self._create_normalizing_flows(base_dist)
        self.neg_prototypes = self._create_normalizing_flows(base_dist)

        # Instance-conditional mapping
        self.ins_map = MLP(configs['in_features']+configs['latent_dim'], configs['latent_dim'],
                           [256, 256], False, nonlinearity="relu",
                           with_output_nonlineartity=False)
        
        # Move model to the right device for consistent initialization
        self.to(configs['device'])
        
        self.reset_parameters()
        
    def reset_parameters(self):
        init_random_seed(self.configs['rand_seed'])
        self.encoder.reset_parameters()
        self.fc_mu.reset_parameters()
        self.fc_logvar.reset_parameters()
        self.label_encoder.reset_parameters()
        self.label_fc_mu.reset_parameters()
        self.label_fc_logvar.reset_parameters()
        self.decoder.reset_parameters()
        self.label_decoder.reset_parameters()
        self.label_classifier.reset_parameters()
        self.ins_map.reset_parameters()
        self.pos_prototypes.reset_parameters()
        self.neg_prototypes.reset_parameters()
    
    def get_config_optim(self):
        return [{'params': self.encoder.parameters()},
                {'params': self.fc_mu.parameters()},
                {'params': self.fc_logvar.parameters()},
                {'params': self.label_encoder.parameters()},
                {'params': self.label_fc_mu.parameters()},
                {'params': self.label_fc_logvar.parameters()},
                {'params': self.pos_prototypes.parameters()},
                {'params': self.neg_prototypes.parameters()},
                {'params': self.decoder.parameters()},
                {'params': self.label_decoder.parameters()},
                {'params': self.label_classifier.parameters()},
                {'params': self.ins_map.parameters()}]
    
    def forward(self, input: Tensor, target: Tensor) -> Tuple[Tensor, ...]:
        # Probabilistic representation of instance and label vector [B x D]
        x_mu, x_logvar = self._encode(input)
        y_mu, y_logvar = self._label_encode(target)
        if self.training:
            z_x = self._reparameterize(x_mu, x_logvar)
            z_y = self._reparameterize(y_mu, y_logvar)
        else:
            z_x = x_mu
            z_y = y_mu
        
        # Latent space regularization
        # KL[q(z|x)||q(z|y)]
        kl_div = torch.mean(0.5*torch.sum(y_logvar-x_logvar-1+torch.exp(x_logvar-y_logvar)
                                          +(y_mu-x_mu)**2/(torch.exp(y_logvar)
                                          +self.configs['eps']), dim=1))
        preds_y = self._label_decode(z_y)
        
        # Instance-conditional mapping for more stable training. Similar technique has been used in Conditional VAE and MPVAE
        z_x = self.ins_map(torch.cat([input, z_x], dim=1))
        
        # Reconstuction
        recons = self._decode(z_x)
        
        # Instance's log probs on positive/negative prototypes of each class label [B x 2 x Q]
        log_ins_class_probs = self._log_density_proto(z_x)
        # Distances between instance and prototypes, i.e. label-specific features [B x 2 x Q]
        # [-KL[q(z|x)||p(z|N^j)], -KL[q(z|x)||p(z|P^j)]] is equivalent to [E_z[p(z|N^j)], E_z[p(z|P^j)]] in implementation
        dists_x = log_ins_class_probs
        
        return input, kl_div, recons, preds_y, dists_x
    
    def training_start(self, train_dataloader):
        '''
        Prepare for training.
        '''
        self.iters_per_epoch = len(train_dataloader)
    
    def loss_function_train(self, preds: Tuple[Tensor, ...], targets: Tensor) -> dict:
        Loss, Recons_loss, Reg_loss, Cls_loss = self._compute_loss(*preds, targets) 

        return {'Loss': Loss,
                'Recons_loss': Recons_loss,
                'Reg_loss': Reg_loss,
                'Cls_loss': Cls_loss}
    
    def predict(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        self.eval()
        with torch.no_grad():
            # Probabilistic representation of instance [B x D]
            x_mu, _ = self._encode(input)
            
            # Instance-conditional mapping for more stable training. Similar technique has been used in Conditional VAE and MPVAE
            z_x = self.ins_map(torch.cat([input, x_mu], dim=1))
            
            # Instance's log probs on positive/negative prototypes of each class label [B x 2 x Q]
            log_class_probs = self._log_density_proto(z_x)
            # Distances between instance and prototypes, i.e. label-specific features [B x 2 x Q]
            # [-KL[q(z|x)||p(z|N^j)], -KL[q(z|x)||p(z|P^j)]] is equivalent to [E_z[p(z|N^j)], E_z[p(z|P^j)]] in implementation
            dists = log_class_probs
            
            # Classification with parameter-free classifiers
            pred_probs = torch.softmax(dists, dim=1)[:, 1, :] # prob for label occurrence
            pred_labels = (pred_probs > 0.5).type_as(pred_probs)
        
        return pred_labels, pred_probs
    
    def configure_optimizers(self) -> Tuple[Any, Any]:
        optimizer = torch.optim.Adam(self.get_config_optim(), lr=self.configs['lr'],
                                     weight_decay=self.configs['weight_decay'])
        if self.configs['lr_scheduler'] == 'step_epoch':
            scheduler = StepLRScheduler(optimizer,
                                        decay_t=self.configs['scheduler_decay_epoch'],
                                        decay_rate=self.configs['scheduler_decay_rate'],
                                        t_in_epoch=True,
                                        iters_per_epoch=self.iters_per_epoch,
                                        warmup_t=self.configs['scheduler_warmup_epoch'])
        else:
            scheduler = None
            
        return optimizer, scheduler
    
    def _create_normalizing_flows(self, base_dist):
        # Create diffeomorphisms
        flow_trans = []
        flow_trans.append(CondNAF(self.configs['latent_dim'], self.configs['num_classes'], [256]))
        flow_trans.append(CondAF(self.configs['latent_dim'], self.configs['num_classes'],
                                 identity_init=True))
        
        return CondTDist(base_dist, flow_trans)
        
    def _encode(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        '''
        Encode the input by passing through the encoder network and return the
        latent codes.

        Parameters
        ----------
        input : Tensor
            Input tensor to encode.

        Returns
        -------
        Tuple(Tensor, Tensor)
            Mean and log variance parameters of the latent Gaussian distribution.
        '''
        result = self.encoder(input)
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)
        
        return mu, logvar
    
    def _label_encode(self, target: Tensor) -> Tuple[Tensor, Tensor]:
        '''
        Encode the input by passing through the encoder network and return the
        latent codes.

        Parameters
        ----------
        target : Tensor
            Input tensor to encode.

        Returns
        -------
        Tuple(Tensor, Tensor)
            Mean and log variance parameters of the latent Gaussian distribution.
        '''
        result = self.label_encoder(target)
        mu = self.label_fc_mu(result)
        logvar = self.label_fc_logvar(result)
        
        return mu, logvar
    
    def _decode(self, z: Tensor) -> Tensor:
        '''
        Decode the latent codes by passing through the decoder network.

        Parameters
        ----------
        z : Tensor [B x D]
            Latent codes to decode.

        Returns
        -------
        Tensor
            Reconstruction.
        '''
        return self.decoder(z)
    
    def _label_decode(self, z: Tensor) -> Tensor:
        '''
        Decode the latent codes by passing through the decoder network.

        Parameters
        ----------
        z : Tensor [B x D]
            Latent codes to decode.

        Returns
        -------
        Tensor
            Reconstruction.
        '''
        return self.label_classifier(self.label_decoder(z))
    
    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        '''
        Reparameterize trick to sample from N(mu, var).

        Parameters
        ----------
        mu : Tensor [B x D]
            Mean of the latent Gaussian.
        logvar : Tensor [B x D]
            Log variance of the latent Gaussian.

        Returns
        -------
        Tensor [B x D]
            Sampled latent codes.
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return eps * std + mu
    
    def _log_density_proto(self, x: Tensor) -> Tensor:
        '''
        Compute instance's log probability on positive/negative prototypes of each class label.

        Parameters
        ----------
        x : Tensor [B x D]
            Point at which density is to be evaluated.

        Returns
        -------
        Tensor [B x 2 x Q]
            log density at x.
        '''
        x_temp = x.unsqueeze(1)
        pos_log_density = self.pos_prototypes.log_prob(x_temp, self.label_encodings) # [B x Q]
        neg_log_density = self.neg_prototypes.log_prob(x_temp, self.label_encodings) # [B x Q]
        
        return torch.stack([neg_log_density, pos_log_density], dim=1)
    
    def _compute_loss(self, input: Tensor, kl_div: Tensor, recons: Tensor,
                      preds_y: Tensor, dists_x: Tensor, targets: Tensor) -> Tuple[Tensor, ...]:
        batch_size = input.size(0)
        
        # Reconstruction loss
        if self.configs['binary_data']:
            Recons_loss = F.binary_cross_entropy_with_logits(recons, input, reduction='sum') / batch_size
        else:
            Recons_loss = F.mse_loss(recons.sigmoid(), input, reduction='sum') / batch_size
        
        # Latent space regularization loss
        Reg_loss = kl_div + F.multilabel_soft_margin_loss(preds_y, targets) * targets.size(1)
        
        # Classification loss
        Cls_loss = F.cross_entropy(dists_x, targets.long()) * targets.size(1)
        
        # Overall loss
        Loss = Recons_loss + self.configs['gamma'] * Reg_loss + self.configs['alpha'] * Cls_loss
        
        return Loss, Recons_loss, Reg_loss, Cls_loss
    def _loss_per_label(self, preds, targets):
        individual_losses = F.multilabel_soft_margin_loss(preds, targets, reduction='none')
        return individual_losses
    def custom_multilabel_soft_margin_loss(self,preds, targets, weights):
#         individual_losses = F.multilabel_soft_margin_loss(preds, targets, reduction='none')
#         return individual_losses
        sigmoid_preds = torch.sigmoid(preds)
        weights_tensor = torch.tensor(weights, dtype=preds.dtype, device=preds.device)
        losses = -weights_tensor * (targets * torch.log(sigmoid_preds + 1e-6) + (1 - targets) * torch.log(1 - sigmoid_preds + 1e-6))
        loss_per_sample = losses.sum(dim=1)
        return loss_per_sample
