class EfficientNet_Segmenter_Config():
	def __init__(
		self,
		input_shape = (512, 512, 3),
		decoder_dims = 256,
		encoder_blocks = 7,
		dropout_ratio = 0.1,
		num_classes = 2,
		output_size = [512, 512],
		decoder_upscale_size = [256, 256] #must be equal to largest output of encoder
	):
		self.input_shape = input_shape
		self.decoder_dims = decoder_dims
		self.encoder_blocks = encoder_blocks
		self.dropout_ratio = dropout_ratio
		self.num_classes = num_classes
		self.output_size = output_size
		self.decoder_upscale_size = decoder_upscale_size