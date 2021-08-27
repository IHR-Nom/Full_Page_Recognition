class Config(object):
    def __init__(self):

        # Learning Rates
        self.lr = 2e-4

        # Epochs
        self.epochs = 200
        self.lr_drop = 20
        self.start_epoch = 0
        self.weight_decay = 1e-4

        # Backbone
        self.backbone = 'resnet34'
        self.position_embedding = 'sine'
        self.dilation = False
        
        # Basic
        self.device = 'cuda'
        self.seed = 42
        self.batch_size = 112
        self.mini_step = 14
        self.num_workers = 8
        self.checkpoint = './checkpoint.pth'
        self.clip_max_norm = 0.1

        # Transformer
        self.hidden_dim = 256
        self.pad_token_id = 0
        self.max_position_embeddings = 128
        self.layer_norm_eps = 1e-12
        self.dropout = 0.1
        self.vocab_size = 30522

        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 1024
        self.nheads = 4
        self.pre_norm = True

        # Dataset
        self.dir = '/data1/hong/datasets/coco'
        self.iam_dir = '/data1/hong/datasets/iam'
        self.wikitext_dir = '/data1/hong/datasets/wikitext'
        self.font_dir = '/data2/mvu/fonts'
        self.limit = -1
        self.max_img_w = 1239
        self.max_img_h = 1078