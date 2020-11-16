class DQN(nn.Module):
  def __init__(self, img_height, img_width):
    super().__init__()
    self.lyr_in = nn.Linear(in_features=1, out_features=24)   
    self.lyr_fc = nn.Linear(in_features=24, out_features=32)
    self.lyr_out = nn.Linear(in_features=32, out_features=2)

  def forward(self, t):
    t = t.flatten(start_dim=1)
    t = F.relu(self.lyr_in(t))
    t = F.relu(self.lyr_fc(t))
    t = self.lyr_out(t)
    return t
