import math
import torch
import gpytorch
import torch.nn as nn

from torch import Tensor
from typing import Callable, Optional

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import to_dense_adj, add_remaining_self_loops, scatter

from gpytorch.models import ExactGP
from gpytorch.means import Mean, ZeroMean
from gpytorch.kernels import Kernel, ScaleKernel, MaternKernel, RBFKernel, PolynomialKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.utils.grid import ScaleToBounds
from gpytorch.likelihoods import (
    Likelihood,
    SoftmaxLikelihood,
    DirichletClassificationLikelihood
)

from gpinfuser.nn import Variational
from gpinfuser.models import (
    SVDKL,
    IDSGP,
    AmortizedSVGP,
    GridInterpolationSVGP
)


# Full GP-Based models
def aggr_conv_matrix(conv_matrix: torch.Tensor, K: int):
    for _ in range(K - 1):
        conv_matrix = torch.sparse.mm(conv_matrix, conv_matrix)
    return conv_matrix

def ggp_norm(edge_index: Tensor, edge_weight: Optional[Tensor]=None):
    num_nodes = maybe_num_nodes(edge_index)

    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, 1.0, num_nodes
    )

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, col, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv = deg.pow_(-1.0)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight

    return edge_index, edge_weight

class GCNKernel(Kernel):
    def __init__(
        self,
        base_kernel: Kernel,
        node_features: Tensor,
        edge_index: Tensor,
        K: int=1,
        norm: str='ggp',
        **kwargs
    ):
        super().__init__(**kwargs)

        self.base_kernel = base_kernel
        self.node_features = node_features
        self.K = K
        self.norm = norm

        if norm == 'gcn':
            edge_index, edge_weight = gcn_norm(edge_index)
        elif norm == 'ggp':
            edge_index, edge_weight = ggp_norm(edge_index)
        else:
            raise ValueError('normalization `%s` is not valid' % (norm,))
        
        self.conv_matrix = to_dense_adj(edge_index, edge_attr=edge_weight)[0]
        self.conv_matrix = self.conv_matrix.to_sparse().to(node_features.device)

        if self.K > 1:
            self.conv_matrix = aggr_conv_matrix(self.conv_matrix, self.K)

    def forward(self, x1: Tensor, x2: Tensor, **kwargs):
        if x1.ndim == 3:
            x1 = x1[0]

        if x2.ndim == 3:
            x2 = x2[0]

        mask_i = x1.squeeze()
        mask_j = x2.squeeze()

        covar = self.base_kernel(self.node_features, **kwargs).evaluate()
        covar = torch.sparse.mm(self.conv_matrix, torch.sparse.mm(self.conv_matrix, covar).t())

        return covar[mask_i, :][:, mask_j]

class GPR(ExactGP):
    def __init__(
        self,
        x_train: Tensor,
        y_train: Tensor,
        mean_module: Mean,
        covar_module: Kernel,
        likelihood: Likelihood
    ):
        super().__init__(x_train, y_train, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x: Tensor, **kwargs):
        return MultivariateNormal(
            mean=self.mean_module(x),
            covariance_matrix=self.covar_module(x, **kwargs)
        )
    
    @torch.inference_mode()
    def predict(self, x: Tensor, **kwargs):
        self.eval()
        f = self(x, **kwargs)
        x = f.sample(torch.Size([1000])).exp()
        p = (x / x.sum(-2, keepdim=True)).mean(0).t()
        return p

def get_ggp(
    x_train: Tensor,
    y_train: Tensor,
    node_features: Tensor,
    edge_index: Tensor,
    num_classes: int,
    norm: str='ggp',
    K: int=1,
    power: float=3.0,
    alpha_epsilon: float=0.01,
    device: str='cuda',
    **kwargs
):
    
    batch_shape = torch.Size([num_classes])
    mean_module = ZeroMean(batch_shape=batch_shape)
    covar_module = ScaleKernel(
        GCNKernel(
            base_kernel=PolynomialKernel(power=power),
            node_features=node_features.to(device),
            edge_index=edge_index,
            K=K,
            norm=norm
        ),
        batch_shape=batch_shape
    )
    likelihood = DirichletClassificationLikelihood(
        targets=y_train,
        alpha_epsilon=alpha_epsilon,
        learn_additional_noise=True
    )

    model = GPR(
        x_train=x_train,
        y_train=likelihood.transformed_targets,
        mean_module=mean_module,
        covar_module=covar_module,
        likelihood=likelihood
    ).to(device)
    parameters = model.parameters()

    return model, parameters

# Neural Networks
class GraphConv(MessagePassing):
    def __init__(
        self,
        channels: int=None,
        ic_alpha: float=0.0,
        im_theta: float=0.0,
        layer: int=None,
        cached: bool=False,
        add_self_loops: bool=True,
        normalize: bool=True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.alpha = ic_alpha
        self.beta = math.log(im_theta / layer + 1)
        if self.beta:
            assert channels is not None
            self.weight = nn.Parameter(torch.empty(channels, channels))
            nn.init.xavier_normal_(self.weight)

        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        super().reset_parameters()

    def forward(
        self,
        x: Tensor,
        x0: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor]=None
    ) -> Tensor:
        
        if self.normalize:
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    num_nodes=x.size(self.node_dim),
                    improved=False,
                    add_self_loops=self.add_self_loops,
                    flow=self.flow,
                    dtype=x.dtype
                )
                if self.cached:
                    self._cached_edge_index = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = cache

        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        if self.alpha:
            x = (1 - self.alpha) * x + self.alpha * x0
        if self.beta:
            x = torch.addmm(x, x, self.weight, beta=1 - self.beta, alpha=self.beta)

        return x
    
    def message(self, x_j: Tensor, edge_weight: Optional[Tensor]) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        
class DeepGCN(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_channels: int,
        num_layers: int,
        ic_alpha: float=0.0,
        im_theta: float=0.0,
        activation: Callable[..., nn.Module]=nn.ReLU(),
        dropout: float=0.0,
        dropout_inputs: bool=True
    ):
        convs = []
        for layer in range(1, num_layers + 1):
            convs.append(
                GraphConv(
                    channels=num_channels,
                    ic_alpha=ic_alpha,
                    im_theta=im_theta,
                    layer=layer,
                    cached=True
                )
            )

        super().__init__()

        self.dropout_inputs = dropout_inputs
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.dense = nn.Linear(in_features, num_channels)
        self.convs = nn.ModuleList(convs)

        nn.init.xavier_normal_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if self.dropout_inputs:
            x = self.dropout(x)

        x = self.dense(x)
        x = self.activation(x)
        x0 = x.clone()

        for conv in self.convs[:-1]:
            x = self.dropout(x)
            x = conv(x, x0, edge_index)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x, x0, edge_index)

        return x
    
class GraphConvFeatureExtractor(nn.Module):
    def __init__(
        self,
        deep_gcn: DeepGCN,
        num_channels: int,
        num_features: int,
        activation: Callable[..., nn.Module]=nn.ReLU(),
        dropout: float=0.0,
    ):
        super().__init__()

        self.deep_gcn = deep_gcn
        self.features = nn.Sequential(
            activation,
            nn.Dropout(dropout),
            nn.Linear(num_channels, num_features)
        )

        nn.init.xavier_normal_(self.features[-1].weight)
        nn.init.zeros_(self.features[-1].bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.deep_gcn(x, edge_index)
        x = self.features(x)
        return x
    
class GNN(GraphConvFeatureExtractor):
    _label_smoothing: float=0.0

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return super().forward(x, edge_index)
    
    @torch.inference_mode()
    def predict(self, x: Tensor, edge_index: Tensor) -> Tensor:
        self.eval()
        return super().forward(x, edge_index).softmax(dim=-1)
    
    def loss(self, raw_logits: Tensor, targets: Tensor) -> Tensor:
        loss = torch.nn.functional.nll_loss(raw_logits.log_softmax(dim=-1), targets)
        if self._label_smoothing:
            smoothing_loss = -raw_logits.mean()
            loss = (1 - self._label_smoothing) * loss + self._label_smoothing * smoothing_loss
        return loss

# GNN-based Gaussian Processes
class SemiSupervisedSVDKL(SVDKL):
    def forward(self, x: Tensor, edge_index: Tensor, mask: Tensor) -> MultivariateNormal:
        out = self.feature_extractor(x, edge_index=edge_index)
        out = self.gplayer.scaler(out[mask])
        out = out.transpose(-1, -2).unsqueeze(-1)
        return self.gplayer(out)
    
    @torch.inference_mode()
    def predict(self, x: Tensor, edge_index: Tensor, mask: Tensor, num_samples: int=1000):
        self.eval()
        with gpytorch.settings.num_likelihood_samples(num_samples):
            targets_dist = self.likelihood(self(x, edge_index=edge_index, mask=mask))
        return targets_dist

class SemiSupervisedIDSGP(IDSGP):
    def forward(self, x: Tensor, edge_index: Tensor, mask: Tensor) -> MultivariateNormal:
        out = self.feature_extractor(x, edge_index=edge_index)
        variational_params = self.variational_estimator(out[mask])
        self.gplayer.set_variational_parameters(*variational_params)
        return self.gplayer(x[mask].unsqueeze(-2))
    
    @torch.inference_mode()
    def predict(self, x: Tensor, edge_index: Tensor, mask: Tensor, num_samples: int=1000):
        self.eval()
        with gpytorch.settings.num_likelihood_samples(num_samples):
            targets_dist = self.likelihood(self(x, edge_index=edge_index, mask=mask))
        return targets_dist

class SemiSupervisedAVDKL(SemiSupervisedIDSGP):
    def forward(self, x: Tensor, edge_index: Tensor, mask: Tensor) -> MultivariateNormal:
        out = self.feature_extractor(x, edge_index=edge_index)
        variational_params = self.variational_estimator(out[mask])
        self.gplayer.set_variational_parameters(*variational_params)
        return self.gplayer(out[mask].unsqueeze(-2))

class DKL(GPR):
    def __init__(
        self,
        x_train: Tensor,
        y_train: Tensor,
        feature_extractor: DeepGCN,
        mean_module: Mean,
        covar_module: Kernel,
        likelihood: DirichletClassificationLikelihood
    ):
        super().__init__(
            x_train=x_train,
            y_train=y_train,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood
        )

        self.feature_extractor = feature_extractor
        self.scaler = ScaleToBounds(-1, 1)
    
    def forward(self, x: Tensor, edge_index: Tensor, features: Tensor):
        mask = x.squeeze()
        features = self.feature_extractor(features, edge_index)
        features = features[mask]
        features = self.scaler(features)
        return super().forward(features)

###
# Models getter
###
def get_svdkl(in_features, num_classes, args, pre_trained_gnn: DeepGCN=None):
    feature_extractor = DeepGCN(
        in_features=in_features,
        num_channels=args.channels,
        num_layers=args.layers,
        ic_alpha=args.alpha,
        im_theta=args.theta,
        dropout=args.dropout
    )

    if pre_trained_gnn is not None:
        feature_extractor.load_state_dict(pre_trained_gnn.state_dict())

    lengthscale_prior = SmoothedBoxPrior(math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp)
    gplayer = GridInterpolationSVGP(
        mean_module=ZeroMean(),
        covar_module=ScaleKernel(MaternKernel(args.matern, lengthscale_prior=lengthscale_prior)),
        num_inducing=args.num_inducing,
        num_tasks=args.channels,
        grid_bounds=[-args.grid_bounds, args.grid_bounds]
    )
    gplayer.covar_module.outputscale = args.outputscale
    likelihood = SoftmaxLikelihood(args.channels, num_classes)

    model = SemiSupervisedSVDKL(feature_extractor, gplayer, likelihood).to(args.device)
    parameters = [
        {'params': model.feature_extractor.dense.parameters(), 'weight_decay': args.dense_weight_decay},
        {'params': model.feature_extractor.convs.parameters(), 'weight_decay': args.convs_weight_decay},
        {'params': model.gplayer.hyperparameters(), 'lr': args.lr * 0.1},
        {'params': model.gplayer.variational_parameters()},
        {'params': model.likelihood.parameters()}
    ]

    if pre_trained_gnn is not None:
        parameters[0]['lr'] = args.lr * 0.1
        parameters[1]['lr'] = args.lr * 0.1

    return model, parameters

def get_idsgp(in_features, num_classes, args, pre_trained_gnn: DeepGCN=None):
    feature_extractor = DeepGCN(
        in_features=in_features,
        num_channels=args.channels,
        num_layers=args.layers,
        ic_alpha=args.alpha,
        im_theta=args.theta,
        dropout=args.dropout
    )

    if pre_trained_gnn is not None:
        feature_extractor.load_state_dict(pre_trained_gnn.state_dict())
    
    variational_estimator = Variational(
        in_features=args.channels,
        num_tasks=num_classes,
        num_features=in_features,
        num_inducing=args.num_inducing,
        saturation=nn.Sequential(nn.Tanh(), nn.Dropout(args.dropout))
    )
    gplayer = AmortizedSVGP(
        mean_module=ZeroMean(),
        covar_module=ScaleKernel(MaternKernel(args.matern)),
        num_inducing=args.num_inducing,
        num_tasks=num_classes
    )
    gplayer.covar_module.outputscale = args.outputscale
    gplayer.covar_module.base_kernel.lengthscale = args.lengthscale

    model = SemiSupervisedIDSGP(
        feature_extractor=feature_extractor,
        variational_estimator=variational_estimator,
        gplayer=gplayer,
        likelihood=SoftmaxLikelihood(num_classes=num_classes, mixing_weights=None)
    ).to(args.device)

    parameters = [
        {'params': model.feature_extractor.dense.parameters(), 'weight_decay': args.dense_weight_decay},
        {'params': model.feature_extractor.convs.parameters(), 'weight_decay': args.convs_weight_decay},
        {'params': model.variational_estimator.parameters(), 'weight_decay': args.dense_weight_decay},
        {'params': model.gplayer.hyperparameters()},
        {'params': model.likelihood.parameters()}
    ]

    if pre_trained_gnn is not None:
        parameters[0]['lr'] = args.lr * 0.1
        parameters[1]['lr'] = args.lr * 0.1

    return model, parameters

def get_avdkl(in_features, num_classes, args, pre_trained_gnn: DeepGCN=None):
    feature_extractor = DeepGCN(
        in_features=in_features,
        num_channels=args.channels,
        num_layers=args.layers,
        ic_alpha=args.alpha,
        im_theta=args.theta,
        dropout=args.dropout
    )

    if pre_trained_gnn is not None:
        feature_extractor.load_state_dict(pre_trained_gnn.state_dict())

    variational_estimator = Variational(
        in_features=args.channels,
        num_features=args.channels,
        num_inducing=args.num_inducing,
        num_tasks=num_classes,
        saturation=nn.Sequential(nn.Tanh(), nn.Dropout(args.dropout))
    )
    gplayer = AmortizedSVGP(
        mean_module=ZeroMean(),
        covar_module=ScaleKernel(MaternKernel(args.matern)),
        num_inducing=args.num_inducing,
        num_tasks=num_classes
    )
    gplayer.covar_module.outputscale = args.outputscale
    gplayer.covar_module.base_kernel.lengthscale = args.lengthscale

    model = SemiSupervisedAVDKL(
        feature_extractor=feature_extractor,
        variational_estimator=variational_estimator,
        gplayer=gplayer,
        likelihood=SoftmaxLikelihood(num_classes=num_classes, mixing_weights=None)
    ).to(args.device)

    parameters = [
        {'params': model.feature_extractor.dense.parameters(), 'weight_decay': args.dense_weight_decay},
        {'params': model.feature_extractor.convs.parameters(), 'weight_decay': args.convs_weight_decay},
        {'params': model.variational_estimator.parameters(), 'weight_decay': args.dense_weight_decay},
        {'params': model.gplayer.hyperparameters(), 'lr': args.lr * 0.1},
        {'params': model.likelihood.parameters()}
    ]

    if pre_trained_gnn is not None:
        parameters[0]['lr'] = args.lr * 0.1
        parameters[1]['lr'] = args.lr * 0.1

    return model, parameters

def get_gnn(in_features, num_classes, args, pre_trained_gnn: DeepGCN=None):
    feature_extractor = DeepGCN(
        in_features=in_features,
        num_channels=args.channels,
        num_layers=args.layers,
        ic_alpha=args.alpha,
        im_theta=args.theta,
        dropout=args.dropout
    )

    if pre_trained_gnn is not None:
        feature_extractor.load_state_dict(pre_trained_gnn.state_dict())
    
    model = GNN(
        deep_gcn=feature_extractor,
        num_channels=args.channels,
        num_features=num_classes,
        dropout=args.dropout
    ).to(args.device)
    parameters = [
        {'params': model.deep_gcn.dense.parameters(), 'weight_decay': args.dense_weight_decay},
        {'params': model.deep_gcn.convs.parameters(), 'weight_decay': args.convs_weight_decay},
        {'params': model.features.parameters(), 'weight_decay': args.dense_weight_decay}
    ]

    if pre_trained_gnn is not None:
        parameters[0]['lr'] = args.lr * 0.1
        parameters[1]['lr'] = args.lr * 0.1

    return model, parameters

# def get_dkl(
#     x_train: Tensor,
#     y_train: Tensor,
#     in_features: int,
#     num_channels: int,
#     num_layers: int,
#     num_classes: int,
#     alpha: float,
#     theta: float,
#     dropout: float,
#     alpha_epsilon: float=0.01,
#     device: str='cuda',
#     pre_trained_gnn: DeepGCN=None,
#     **kwargs
# ):
#     feature_extractor = DeepGCN(
#         in_features=in_features,
#         num_channels=num_channels,
#         num_layers=num_layers,
#         ic_alpha=alpha,
#         im_theta=theta,
#         dropout=dropout
#     )

#     if pre_trained_gnn is not None:
#         feature_extractor.load_state_dict(pre_trained_gnn.state_dict())

#     batch_shape = torch.Size([num_classes])
#     likelihood = DirichletClassificationLikelihood(
#         targets=y_train,
#         alpha_epsilon=alpha_epsilon,
#         learn_additional_noise=True
#     )

#     model = DKL(
#         x_train=x_train,
#         y_train=likelihood.transformed_targets,
#         feature_extractor=feature_extractor,
#         mean_module=ZeroMean(batch_shape=batch_shape),
#         covar_module=ScaleKernel(
#             base_kernel=RBFKernel(ard_num_dims=num_channels),
#             batch_shape=batch_shape
#         ),
#         likelihood=likelihood
#     ).to(device)

#     parameters = [
#         {'params': model.feature_extractor.dense.parameters(), 'weight_decay': DENSE_WEIGHT_DECAY},
#         {'params': model.feature_extractor.convs.parameters(), 'weight_decay': CONVS_WEIGHT_DECAY},
#         {'params': model.mean_module.parameters()},
#         {'params': model.covar_module.parameters()},
#         {'params': model.likelihood.parameters()},
#     ]

#     if pre_trained_gnn is not None:
#         parameters[0]['lr'] = 5e-4

#     return model, parameters
