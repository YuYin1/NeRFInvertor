import math

FACES_finetune = {
    'global': {
        'img_size': 256,
        'batch_size': 1, # batchsize per GPU. We use 8 GPUs by default so that the effective batchsize for an iteration is 4*8=32
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-5,
        'disc_lr': 2e-4,
        'betas': (0, 0.9),
        'grad_clip': 1.,
    },
    'process': {
        'class': 'Gan3DProcess',
        'kwargs': {
            'batch_split': 4,
            'real_pos_lambda': 15.,
            'r1_lambda': 1.,
            'pos_lambda': 15.,
        }
    },
    'generator': {
        'class': 'GramGenerator',
        'kwargs': {
            'z_dim': 256,
            'img_size': 256,
            'h_stddev': 0.3,
            'v_stddev': 0.155,
            'h_mean': math.pi*0.5,
            'v_mean': math.pi*0.5,
            'sample_dist': 'gaussian',
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 128,
                'layer_num_sample': 3,
                'center': (0, 0, -1.5),
                'init_radius': 0,
            },
        },
        'renderer': {
            'class': 'manifold_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 24,
                'levels_start': 23,
                'levels_end': 8,
                'last_back': False,
                'white_back': False,
                'background': True,
            }
        }
    },
    'dataset': {
        'class': 'FACES_finetune',
        'kwargs': {
            'img_size': 256,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 12,
        'ray_start': 0.88,
        'ray_end': 1.12,
    }
}


CATS_finetune = {
    'global': {
        'img_size': 256,
        'batch_size': 1,
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-5,
        'disc_lr': 2e-4,
        'betas': (0, 0.9),
        'grad_clip': 1.,
    },
    'process': {
        'class': 'Gan3DProcess',
        'kwargs': {
            'batch_split': 2,
            'real_pos_lambda': 30.,
            'r1_lambda': 1.,
            'pos_lambda': 15.,
        }
    },
    'generator': {
        'class': 'GramGenerator',
        'kwargs': {
            'z_dim': 256,
            'img_size': 256,
            'h_stddev': 0.3,
            'v_stddev': 0.155,
            'h_mean': math.pi*0.5,
            'v_mean': math.pi*0.5,
            'sample_dist': 'gaussian',
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 64,
                'layer_num_sample': 3,
                'center': (0, 0, -1.5),
                'init_radius': 0,
            },
        },
        'renderer': {
            'class': 'manifold_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 24,
                'levels_start': 23,
                'levels_end': 8,
                'last_back': False,
                'white_back': False,
                'background': True,
            }
        }
    },
    'discriminator': {
        'class': 'GramDiscriminator',
        'kwargs': {
            'img_size': 256,
        }
    },
    'dataset': {
        'class': 'CATS_finetune',
        'kwargs': {
            'img_size': 256,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 12,
        'ray_start': 0.88,
        'ray_end': 1.12,
    }
}


FACES_default = {
    'global': {
        'img_size': 256,
        'batch_size': 1, # batchsize per GPU. We use 8 GPUs by default so that the effective batchsize for an iteration is 4*8=32
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-5,
        'disc_lr': 2e-4,
        'betas': (0, 0.9),
        'grad_clip': 1.,
    },
    'process': {
        'class': 'Gan3DProcess',
        'kwargs': {
            'batch_split': 4,
            'real_pos_lambda': 15.,
            'r1_lambda': 1.,
            'pos_lambda': 15.,
        }
    },
    'generator': {
        'class': 'GramGenerator',
        'kwargs': {
            'z_dim': 256,
            'img_size': 256,
            'h_stddev': 0.3,
            'v_stddev': 0.155,
            'h_mean': math.pi*0.5,
            'v_mean': math.pi*0.5,
            'sample_dist': 'gaussian',
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 128,
                'layer_num_sample': 3,
                'center': (0, 0, -1.5),
                'init_radius': 0,
            },
        },
        'renderer': {
            'class': 'manifold_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 24,
                'levels_start': 23,
                'levels_end': 8,
                'last_back': False,
                'white_back': False,
                'background': True,
            }
        }
    },
    'dataset': {
        'class': 'FFHQ',
        'kwargs': {
            'img_size': 256,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 12,
        'ray_start': 0.88,
        'ray_end': 1.12,
    }
}

CATS_default = {
    'global': {
        'img_size': 256,
        'batch_size': 1,
        'z_dist': 'gaussian',
    },
    'optimizer': {
        'gen_lr': 2e-5,
        'disc_lr': 2e-4,
        'betas': (0, 0.9),
        'grad_clip': 1.,
    },
    'process': {
        'class': 'Gan3DProcess',
        'kwargs': {
            'batch_split': 2,
            'real_pos_lambda': 30.,
            'r1_lambda': 1.,
            'pos_lambda': 15.,
        }
    },
    'generator': {
        'class': 'GramGenerator',
        'kwargs': {
            'z_dim': 256,
            'img_size': 256,
            'h_stddev': 0.3,
            'v_stddev': 0.155,
            'h_mean': math.pi*0.5,
            'v_mean': math.pi*0.5,
            'sample_dist': 'gaussian',
        },
        'representation': {
            'class': 'gram',
            'kwargs': {
                'hidden_dim': 256,
                'sigma_clamp_mode': 'softplus',
                'rgb_clamp_mode': 'widen_sigmoid',
                'hidden_dim_sample': 64,
                'layer_num_sample': 3,
                'center': (0, 0, -1.5),
                'init_radius': 0,
            },
        },
        'renderer': {
            'class': 'manifold_renderer',
            'kwargs': {
                'num_samples': 64,
                'num_manifolds': 24,
                'levels_start': 23,
                'levels_end': 8,
                'last_back': False,
                'white_back': False,
                'background': True,
            }
        }
    },
    'discriminator': {
        'class': 'GramDiscriminator',
        'kwargs': {
            'img_size': 256,
        }
    },
    'dataset': {
        'class': 'CATS',
        'kwargs': {
            'img_size': 256,
            'real_pose': True,
        }
    },
    'camera': {
        'fov': 12,
        'ray_start': 0.88,
        'ray_end': 1.12,
    }
}

