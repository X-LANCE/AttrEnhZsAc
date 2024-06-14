import torch
import torch.nn as nn
from models import audio_models
from models import text_models
from abc import abstractmethod
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim=1280, out_dim=256, n_hidden=1):
        super(MLP, self).__init__() 
        mlp = nn.ModuleList()
        _in_dim = in_dim
        for _ in range(n_hidden):
            next_dim = _in_dim // 2\
                if _in_dim >= 2 * out_dim else out_dim
            mlp.append(nn.Linear(_in_dim, next_dim))
            # mlp.append(nn.BatchNorm1d(next_dim))
            # mlp.append(nn.ReLU())
            # mlp.append(nn.Dropout(p=0.1))
            _in_dim = next_dim
        mlp.append(nn.Linear(_in_dim, out_dim))
        
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)


class PretrainVectorBilinearModel(nn.Module):
    """
    model for Baseline pretraining audio-text embeddings
    Training: (BxD) @ (DxT) = (BxT), where T is the number of classes
    Inference: (BxD) @ (DxT) = (BxT)
    ========================
    Initialization Parameters:
    audio_embed_dim (int): dimension of audio embeddings
    text_embeddings (torch.tensor): T x D tensor of text embeddings
    use_MLP (bool): whether to use MLP mapping, DEFAULT: False
    latent_dim (int): dimension of latent space, DEFAULT: None
    """
    def __init__(self,
                 audio_embed_dim,
                 text_embeddings,
                 use_MLP=False,
                 latent_dim=None):
        super(PretrainVectorBilinearModel, self).__init__()
        self.text_embeds = text_embeddings
        text_embed_dim = text_embeddings.shape[-1]
        self.latent_dim = latent_dim
        self.use_MLP = use_MLP
        if self.latent_dim is None:
            if use_MLP:
                self.audio_proj = MLP(audio_embed_dim, text_embed_dim, n_hidden=1)
                print("MLP mapping")
            else:
                self.W = nn.Parameter(torch.randn(audio_embed_dim, text_embed_dim) * 0.01)
                print("Linear mapping")
        else:
            self.U = nn.Parameter(torch.randn(audio_embed_dim, latent_dim) * 0.01)
            self.V = nn.Parameter(torch.randn(latent_dim, text_embed_dim) * 0.01)
            self.activate_fn = nn.Tanh()
            print("Nonlinear mapping")

    def forward(self, audio_embeds, targets):
        """
        audio_embeds: B x D tensor of audio embeddings
        """
        if self.latent_dim is None:
            if self.use_MLP:
                scores = self.audio_proj(audio_embeds) @ self.text_embeds.T
                return {
                    'score': scores,
                    'target': targets,
                }
            else:
                scores = audio_embeds @ self.W @ self.text_embeds.T # (B, T), B: batch_size, T: number of class
                return {
                    'W': self.W,
                    'score': scores,
                    'target': targets,
                }
        else:
            scores = self.activate_fn(audio_embeds @ self.U) @ self.V @ self.text_embeds.T
            return {
                'W': [self.U, self.V],
                'score': scores,
                'target': targets,
            }

    @torch.no_grad()
    def calculate_score(self, audio_embeds, targets):
        model_out = self.forward(audio_embeds, targets)
        return {
            'score': model_out['score'],
            'target': targets,
        }
    

class PretrainRandomAttrVectorModel(nn.Module):
    """
    model for Proposed pretraining audio-text embeddings
    Training: (BxD) @ (DxB) = (BxB), where B is the batch size
    Inference: (BxD) @ (DxT) = (BxT), where T is the number of classes
    ========================
    Initialization Parameters:
    text_arch (str): text model architecture (RoBERTa, etc.)
    text_pretrain (str): text model pretrain (sentence-transformers/all-MiniLM-L6-v2, etc.)
    audio_embed_dim (int): dimension of audio embeddings
    """
    def __init__(self,
                 text_model_kwargs: dict,
                 audio_embed_dim: int):
        super(PretrainRandomAttrVectorModel, self).__init__()
        text_embedding_type = text_model_kwargs['text_embedding_type']
        text_embedding_model = text_model_kwargs['text_embedding_model']
        self.text_encoder = getattr(text_models, text_embedding_type)(text_embedding_model=text_embedding_model)
        # self.audio_proj = MLP(audio_embed_dim, self.text_encoder.embed_dim, n_hidden=1)
        # Modify on 2024/06/06, one linear layer
        self.audio_proj = MLP(audio_embed_dim, self.text_encoder.embed_dim, n_hidden=0)
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_text(self, text):
        """
        Encode text to text embeddings with fixed text encoder
        
        ========================
        Parameters:
        text (dict): with keys 'input_ids' and 'attention_mask'

        ========================
        Returns:
        text_embeds (torch.tensor): B x D tensor of text embeddings
        """
        text_embeds = self.text_encoder(**text)['mean_emb']
        return text_embeds
    
    def encode_audio(self, audio_embed):
        audio_proj = self.audio_proj(audio_embed)
        return audio_proj


    def forward(self, audio_embeds, text, targets):
        """
        Encode audio and text to same space and calculate scores
        
        ========================
        Parameters:
        audio_embeds (torch.tensor): B x T tensor of audio embeddings
        text (dict): with keys 'input_ids' and 'attention_mask'
        targets (torch.tensor): B tensor of target labels
        """
        audio_proj = self.encode_audio(audio_embeds)
        audio_proj = F.normalize(audio_proj, dim=-1)
        text_proj = self.encode_text(text)
        text_proj = F.normalize(text_proj, dim=-1)

        return {
            'audio_proj': audio_proj,
            'text_proj': text_proj,
            'audio_target': targets,
            'text_target': targets
        }


class AudioTrainModel(nn.Module):
    """
    model for audio classification
    """
    def __init__(self,
                 audio_arch: str,
                 audio_model_kwargs: dict,
                 audio_pretrain: str = None,
                 n_class: int = 50,
                 **kwargs):
        super(AudioTrainModel, self).__init__()
        self.audio_encoder = getattr(audio_models, audio_arch)(**audio_model_kwargs)
        if audio_pretrain: # load pretrained parameters without output fc layer
            params = torch.load(audio_pretrain)['model']
            params = {key: value for key, value in params.items() if "fc_audioset" not in key}
            self.audio_encoder.load_state_dict(params)
            print('Load Audioset-pretrained model')
        with torch.no_grad():
            if 'Cnn' or 'ASTModel' in audio_arch:
                n_seconds = audio_model_kwargs.get('n_seconds', 10)
                sample_rate = audio_model_kwargs.get('sample_rate', 32000)
                waveform = torch.zeros(2, n_seconds * sample_rate)
            else: 
                raise NotImplementedError
            audio_embed_dim = self.audio_encoder(waveform)['embedding'].shape[-1]
        self.fc = nn.Linear(audio_embed_dim, n_class)
        
    def encode_audio(self, waveform):
        """
        Encode waveform to audio embeddings
        """
        audio_embed = self.audio_encoder(waveform)['embedding']
        return audio_embed

    def forward(self, data, targets):
        """

        ========================
        Parameters:
        data (torch.tensor): B x D waveform data
        targets (torch.tensor): B tensor of target labels
        """
        audio_embed = self.encode_audio(data)
        fc_out = self.fc(audio_embed)
        return {
            'logit': fc_out,
            'target': targets,
        }

