
def dbitseq_config(sequence_length,
            categories,
            task='unsupervised',
            batch_size=64,
            hidden_dim=64,
            train_epochs=100,
            lr=1e-4,
            n_patches=8,
            n_transformers=1,
            n_head=4,
            n_routers=1,
            n_experts=16,
            gate_loss_weight=1e-1,
            decoder_loss_weight=1e-1,
            top_k=2,
            patch=True,
            **kwargs):

    dyngen_config = {
        "task": task, # unsupervised, supervised
        "train_batch_size": batch_size,
        "train_epochs": train_epochs,
        'train_loss_weight': None,
        "lr": lr, # deafult: 1e-3
        "checkpoint": 50,
        # "fuser_type": "SeqConcatFuser",
        "fuser_type": "AttentionFuser",
        "noise_level": None,
        "categories": categories,
        "sequences_length": sequence_length,
        "hidden_dim": hidden_dim,
        # "sequences_length": [35, 22, 35],
        "gate_loss_weight": gate_loss_weight, # {0.01, 0.001, 0.01} default: 1e-1
        "decoder_loss_weight": decoder_loss_weight,
        "multimodality": {
            "num_experts": 16,
            "base_capacity": 16,
            "capacity_per_expert": 655,
            "num_tasks": 1,
            "load_expert_count": False,
            "seed": 1,
            "attn_modality_specific": False,
            "modalities_name": ['0'],
            "modality_remap": {},
            "capacity_ratio": 1.0,
            "dynamic_reweight": False,
            "top_k": 2,
            "attn_top_k": 2,
        },
        "encoders" : 
            [
            ## Patch
                {
                "type": "PatchEmbeddings",
                "feature_size": sequence_length[i],
                "num_patches": n_patches,  # {1, 8, 16, 64}
                "embed_dim": hidden_dim,
                } for i in range(len(sequence_length))
            ] if patch else
            [
            ## OG
                {
                "type": "EncoderMLP",
                "input": 1,
                "output": hidden_dim,
                "activation": "relu",
                "use_batch_norm": False,
                "use_layer_norm": True,
                # "patching_size": 16,
                # "stride": 16
                } for _ in range(len(sequence_length))
            ],
        "transformer": [
            {
                "type": "TransformerEncoderLayer",
                "d_model": hidden_dim,
                "nhead": n_head,
                "mlp_sparse": True if i % 2 == 0 else False,
                # "mlp_sparse": True,
                "gate": 'AddtionalNoisyGate',
                "n_router": n_routers,
                "self_attn": True,
                "num_expert": n_experts # {4, 8, 16, 24, 32} - higher lr, num_epochs
                # use 2, 3 layers of Transformer
            } for i in range(n_transformers)
            

        ],
        "decoders": [
            {
                "type": "DecoderMLP",
                "in_features": hidden_dim,
                "hidden_features": hidden_dim,
                "out_features": sequence_length[i]
            } for i in range(len(sequence_length))
        ]
    }

    return dyngen_config

