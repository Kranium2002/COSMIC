import torch

from cosmic import Cosmic


def test_optimizer_step_decreases_loss():
    """Test that Cosmic optimizer actually decreases loss."""
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = Cosmic(model.parameters(), lr=0.1, weight_decay=0.0)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[1.0], [2.0]])

    optimizer.zero_grad()
    loss_before = torch.nn.functional.mse_loss(model(x), y)
    loss_before.backward()
    optimizer.step()

    loss_after = torch.nn.functional.mse_loss(model(x), y)

    assert loss_after < loss_before, f"Loss should decrease: {loss_before} -> {loss_after}"
