class CNN2D(nn.Module):
	def __init__(self):
		# F_temp = 8
		# k_temp = 128
		# F_spac = 2 * F_temp
		# k_temp = 64

		F1 = 8
		D = 2
		k_temp = 128
		C = 64
		F2 = F1 * D

		super(CNN2D, self).__init__()

		# self.conv = nn.Sequential(
		# 	nn.Conv2d(1, F1, (1, k_temp)),
		# 	nn.BatchNorm2d(2*F1),
		# 	nn.Conv2d(F1, D*F1, (C, 1)),
		# 	nn.BatchNorm2d(2*D*F1),
		# 	nn.ReLu(),
		# 	nn.AvgPool2d((1, 4)),
		# 	nn.Dropout(p=0.5),
		# 	nn.Conv2D(D*F1, F2, (1, 16)),
		# 	nn.BatchNorm2d(2*F2),
		# 	nn.ReLu(),
		# 	nn.AvgPool2d((1, 8)),
		# 	nn.Dropout(p=0.5),
		# 	)
		# self.fc = nn.Linear(
		# 	nn.flatten(),
		# 	nn.Softmax())

		self.conv_temp = nn.Sequential(
			nn.Conv2d(1, 8, (1, 128)),
			nn.BatchNorm2d(8)
			)

		self.conv_spac = nn.Sequential(
			nn.Conv2d(8, 16, (64, 1)),
			nn.BatchNorm2d(16),
			nn.ReLu(),
			nn.AvgPool2d(1, 4),
			nn.Dropout(p=0.5)
			)

		self.conv_point = nn.Sequential(
			nn.Conv2d(16, 16, (1, 16)),
			nn.Conv2d(16, 16, (1, 1))
			nn.BatchNorm2d(16),
			nn.ReLu(),
			nn.AvgPool2d((1, 8)),
			nn.Dropout(p=0.5)
			)

		self.out = nn.Linear((16 * 8), 2)

	def forward(self, x):
		x = self.conv_temp(x)
		x = self.conv_spac(x)
		x = self.conv_point(x)

		x = self.out(x)
		return nn.Softmax(x)
		return x