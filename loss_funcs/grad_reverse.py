from torch.autograd import Function


class GRL(Function):
    @staticmethod
    def forward(ctx, input, lambda_):
        ctx.lambda_ = lambda_
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.lambda_

        return grad_input, None


def grad_reverse(x, lambd=-1.0):
    return GRL.apply(x, lambd)