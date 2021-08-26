class Config(object):
    def __init__(self):

        # Learning Rates
        self.lr = 5e-4

        # Epochs
        self.epochs = 200
        self.lr_drop = 20
        self.start_epoch = 0
        self.weight_decay = 1e-4

        # Backbone
        self.backbone = 'resnet50'
        self.position_embedding = 'sine'
        self.dilation = True
        
        # Basic
        self.device = 'cuda'
        self.seed = 42
        self.batch_size = 20
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
        self.dim_feedforward = 2048
        self.nheads = 8
        self.pre_norm = True

        # Dataset
        self.dir = '/data1/hong/datasets/coco'
        self.iam_dir = '/data1/hong/datasets/iam'
        self.wikitext_dir = '/data1/hong/datasets/wikitext'
        self.font_dir = '/data2/mvu/fonts'
        self.limit = -1
        self.max_img_w = 1239
        self.max_img_h = 1078

        # ChuNom Dataset
        self.chunom_max_img_h = 495
        self.chunom_max_img_w = 763
        self.chunom_max_patch_h = 46
        self.chunom_max_patch_w = 350
        self.chunom_dir = '/data1/hong/datasets/chunom/handwritten/pages'
        self.synthetic_chunom_dir = '/data1/hong/datasets/chunom/handwritten/patches'
        self.generated_chunom_dir = '/data1/hong/datasets/chunom/handwritten/nlp/nlp_data.csv'
        self.chunom_font_dir = 'data1/hong/nom_fonts'