class GRU(nn.Module):
  def __init__(self, params):
    super(GRU, self).__init__()

    self.gru_layer = nn.GRU(
        params["input_size"],
        params["hidden_size"],
        params["num_layers"],
        bias=params["gru"]["bias"],
        batch_first=True,
        bidirectional=params["gru"]["bidirectional"],
        dropout=params["gru"]["dropout"],
        device=device
    )
    self.out = nn.Linear(params["hidden_size"]*(2 if params["gru"]["bidirectional"] else 1), params["num_classes"])

  def forward(self, x):
    r_out, t = self.gru_layer(x, None)
    test_output = self.out(r_out[:,-1,:])
    return test_output