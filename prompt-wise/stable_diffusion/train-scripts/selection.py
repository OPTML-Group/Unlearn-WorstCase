import torch
import wandb
import utils
import numpy as np

from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler


def bisection(a, eps, xi=1e-5, ub=1, max_iter=1e2):
    with torch.no_grad():

        def value(a, x):
            return torch.sum(torch.clamp(a - x, 0, ub)) - eps

        lef = torch.min(a - 1)
        sign = torch.sign(value(a, lef))
        rig = torch.max(a)

        for _ in range(int(max_iter)):
            mid = (lef + rig) / 2
            vm = value(a, mid)
            if torch.abs(vm) < xi:
                break
            if torch.sign(vm) == sign:
                lef = mid
            else:
                rig = mid

        result = torch.clamp(a - mid, 0, ub)

    return result


class DataSelection:
    def init_models(self, config_path, ckpt_path):
        self.theta_o_state_dict = utils.load_state_dict(ckpt_path)
        self.theta_o = utils.init_model_from_config(
            config_path, self.device_o, self.theta_o_state_dict
        )
        self.theta_u = utils.init_model_from_config(config_path, self.device_u)
        self.theta_o_sampler = DDIMSampler(self.theta_o)
        self.theta_u_sampler = DDIMSampler(self.theta_u)

    def reset_theta_u(self):
        m, u = self.theta_u.load_state_dict(self.theta_o_state_dict, strict=False)
        self.theta_u.to(self.device_u)

    def __init__(
        self,
        prompts,
        config_path,
        ckpt_path,
        devices,
        unlearn_method,
        eval_method,
        w_lr,
        gamma,
        ratio=0.1,
        wandb=wandb,
        save_path=None,
    ):
        self.prompts = prompts
        self.w_lr = w_lr
        self.gamma = gamma
        self.num_indexes_to_replace = int(len(prompts) * ratio)

        self.unlearn_method = unlearn_method
        self.eval_method = eval_method

        self.device_o = devices[0]
        self.device_u = devices[1]

        self.init_models(config_path, ckpt_path)
        self.wandb = wandb
        self.save_path = save_path

    def unlearn(self, w):
        self.reset_theta_u()
        self.unlearn_method(
            self.theta_o,
            self.theta_u,
            self.theta_u_sampler,
            self.prompts,
            w,
        )

    def eval(self):
        return self.eval_method.eval(
            self.theta_u,
            self.prompts,
        )

    def step(self, w):
        self.unlearn(w)
        w_grad = self.eval()
        w -= self.w_lr * (w_grad + self.gamma * 2 * w)
        w = bisection(w, self.num_indexes_to_replace)

        loss = torch.sum(w * w_grad)
        if self.wandb is not None:
            self.wandb.log({"upper_loss": loss.item()})
            self.wandb.log({"w": w.cpu().numpy()})
        return w, loss

    def optimize(self):
        w = (
            torch.ones(len(self.prompts))
            * self.num_indexes_to_replace
            / len(self.prompts)
        )  # initialize the weights

        for i in range(10):
            print(f"Iteraton #{i}")
            w, loss = self.step(w)
            print("Loss: ", loss)
            np.save(f"{self.save_path}/select_weight_{i}.npy", w.detach().cpu().numpy())
        return w