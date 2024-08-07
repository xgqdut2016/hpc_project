# ---------------------------------------------------------------------------------------------------
# BFGS Optimizer
# The code is modified from torch.optim.LBFGS
# ---------------------------------------------------------------------------------------------------
import torch
from functools import reduce
from torch.optim.optimizer import Optimizer
import linesearch

class BFGS(Optimizer):
    """
    Arguments:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """
    def __init__(self,
                 params,
                 lr=1,
                 max_iter=20,
                 max_eval=None,
                 tolerance_grad=1e-5,
                 tolerance_change=1e-9,
                 history_size=100,
                 line_search_fn=None):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn)
        super(BFGS, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("BFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache
    
    def _clone_param(self):
        return [p.clone() for p in self._params]

    def _set_param(self, params_data):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.copy_(params_data[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._numel()

    def _flatten(self, x):
        views = []
        for p in x:
            view = p.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_grad_flat(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._numel()

    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']

        # NOTE: BFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        loss = float(closure())
        grad_flat = self._gather_grad_flat()
        current_evals = 1
        state['func_evals'] += 1

        # optimal condition
        opt_cond = grad_flat.abs().max() <= tolerance_grad
        if opt_cond:
            return closure()

        # tensors cached in state (for tracing)
        x_prev = self._clone_param()
        x_prev = self._flatten(x_prev)
        
        direction = state.get('direction')
        step_size = state.get('step_size')
        loss_prev = state.get('loss_prev')
        loss_prev_prev = state.get('loss_prev_prev')
        grad_prev_flat = state.get('grad_prev_flat')
        Hess_inv = state.get('Hess_inv')

        if loss_prev is None:
            loss_prev = loss
        if grad_prev_flat is None:
            grad_prev_flat = grad_flat.clone()
        if loss_prev_prev is None:
            loss_prev_prev = loss_prev + ((grad_prev_flat**2).sum())**0.5 / 2
        if Hess_inv is None:
            Hess_inv = torch.eye(self._numel(), dtype=self._params[0].dtype).to(self._params[0].device)

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            direction = torch.mm(Hess_inv, grad_flat.neg().view(-1,1)).view(-1)

            # directional derivative
            grad_dot_dir = grad_flat.dot(direction)
            if grad_dot_dir > -tolerance_change:
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn != "strong_wolfe":
                    raise RuntimeError("only 'strong_wolfe' is supported")
                else:
                    def obj_func_loss(x):
                        self._set_param(x)
                        return float(closure())

                    def obj_func_grad(x):
                        # self._set_param(x)
                        # loss = float(closure())
                        return self._gather_grad_flat()
                    try:
                        step_size, ls_func_evals, _, loss_prev, loss_prev_prev, _ = \
                             linesearch._line_search_wolfe12(obj_func_loss, obj_func_grad,
                                                            x_prev, direction, grad_flat,
                                                            loss_prev, loss_prev_prev,
                                                            amin=1e-100, amax=1e100)
                    except:
                        self._set_param(x_prev)
                        loss_prev = loss
                        loss_prev_prev = loss_prev + ((grad_prev_flat**2).sum())**0.5 / 2
                        grad_flat = self._gather_grad_flat()
                        Hess_inv = torch.eye(self._numel(), dtype=self._params[0].dtype).to(self._params[0].device)
                        direction = torch.mm(Hess_inv, grad_flat.neg().view(-1,1)).view(-1)

                        current_evals += 1
                        state['func_evals'] += 1
                        break
                        '''
                        step_size, ls_func_evals, _, loss_prev, loss_prev_prev, _ = \
                             linesearch._line_search_wolfe12(obj_func_loss, obj_func_grad,
                                                             x_prev, direction, grad_flat,
                                                             loss_prev, loss_prev_prev,
                                                             amin=1e-100, amax=1e100)
                        '''
            else:
                step_size = lr
                x = x_prev + step_size*direction
                loss_prev_prev = loss_prev
                loss_prev = float(closure())
                ls_func_evals = 1

            x = x_prev + step_size*direction
            s = direction.mul(step_size)

            grad_flat = self._gather_grad_flat()
            opt_cond = grad_flat.abs().max() <= tolerance_grad
            y = grad_flat.sub(grad_prev_flat)
            
            ys = y.dot(s)
            Hess_inv = Hess_inv - torch.mm(s.view(-1,1)/ys, torch.mm(y.view(-1,1).T, Hess_inv))
            Hess_inv = Hess_inv - torch.mm(torch.mm(Hess_inv, y.view(-1,1)), s.view(-1,1).T/ys)
            Hess_inv = Hess_inv + torch.mm(s.view(-1,1), s.view(-1,1).T) / ys

            x_prev = x
            grad_prev_flat = grad_flat
            
            # keep track of nb of iterations
            n_iter += 1
            state['n_iter'] += 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals
            
            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
            if direction.mul(step_size).abs().max() <= tolerance_change:
                break

            if abs(loss_prev - loss_prev_prev) < tolerance_change:
                break

        state['direction'] = direction
        state['step_size'] = step_size
        state['grad_prev_flat'] = grad_prev_flat
        state['loss_prev'] = loss_prev
        state['loss_prev_prev'] = loss_prev_prev
        state['Hess_inv'] = Hess_inv

        return closure()



