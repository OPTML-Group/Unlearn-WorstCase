import torch
from tqdm import tqdm

import wandb
import numpy as np


import os
from PIL import Image

from einops import rearrange, repeat
from torchvision.utils import make_grid
from diffusers import LMSDiscreteScheduler


@torch.no_grad()
def sample_model(
    model,
    sampler,
    c,
    h,
    w,
    ddim_steps,
    scale,
    ddim_eta,
    start_code=None,
    n_samples=1,
    t_start=-1,
    log_every_t=None,
    till_T=None,
    verbose=True,
):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(
        S=ddim_steps,
        conditioning=c,
        batch_size=n_samples,
        shape=shape,
        verbose=False,
        x_T=start_code,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=uc,
        eta=ddim_eta,
        verbose_iter=verbose,
        t_start=t_start,
        log_every_t=log_t,
        till_T=till_T,
    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim


class ESD:
    def get_esd_parameters(self, theta_u, train_method):
        # choose parameters to train based on train_method
        parameters = []
        for name, param in theta_u.model.diffusion_model.named_parameters():
            # train all layers except x-attns and time_embed layers
            if train_method == "noxattn":
                if name.startswith("out.") or "attn2" in name or "time_embed" in name:
                    pass
                else:
                    parameters.append(param)
            # train only self attention layers
            if train_method == "selfattn":
                if "attn1" in name:
                    # print(name)
                    parameters.append(param)
            # train only x attention layers
            if train_method == "xattn":
                if "attn2" in name:
                    # print(name)
                    parameters.append(param)
            # train all layers
            if train_method == "full":
                # print(name)
                parameters.append(param)
            # train all layers except time embed layers
            if train_method == "notime":
                if not (name.startswith("out.") or "time_embed" in name):
                    # print(name)
                    parameters.append(param)
            if train_method == "xlayer":
                if "attn2" in name:
                    if "output_blocks.6." in name or "output_blocks.8." in name:
                        # print(name)
                        parameters.append(param)
            if train_method == "selflayer":
                if "attn1" in name:
                    if "input_blocks.4." in name or "input_blocks.7." in name:
                        # print(name)
                        parameters.append(param)
        return parameters

    def get_optimizer(self, parameters):
        return torch.optim.Adam(parameters, lr=self.lr)

    def __init__(
        self,
        train_method,
        start_guidance,
        negative_guidance,
        iterations,
        eval_iters,
        lr,
        output_name,
        image_size=512,
        ddim_steps=50,
        wandb=wandb,
    ):
        self.train_method = train_method
        self.start_guidance = start_guidance
        self.negative_guidance = negative_guidance
        self.iterations = iterations
        self.eval_iters = eval_iters
        self.lr = lr
        self.output_name = output_name
        self.image_size = image_size
        self.ddim_steps = ddim_steps
        self.wandb = wandb

    def __call__(
        self,
        theta_o,
        theta_u,
        sampler_u,
        prompts,
        w=None,
    ):
        if w is None:
            w = torch.ones(len(prompts))

        self.train_esd(
            theta_o=theta_o,
            theta_u=theta_u,
            sampler_u=sampler_u,
            prompts=prompts,
            w=w,
        )

    def train_esd(self, theta_o, theta_u, sampler_u, prompts, w):
        print("Unlearn prompts:", prompts)

        def prompt_generator(prompts):
            while True:
                for p in zip(prompts, w):
                    yield p

        ddim_eta = 0
        # set model to train
        theta_u.train()
        # create a lambda function for cleaner use of sampling code (only denoising till time step t)
        quick_sample_till_t = lambda x, s, code, t: sample_model(
            theta_u,
            sampler_u,
            x,
            self.image_size,
            self.image_size,
            self.ddim_steps,
            s,
            ddim_eta,
            start_code=code,
            till_T=t,
            verbose=False,
        )
        parameters = self.get_esd_parameters(theta_u, self.train_method)
        opt = self.get_optimizer(parameters)

        criteria = torch.nn.MSELoss()

        pbar = tqdm(range(self.iterations))
        prompt_gen = prompt_generator(prompts)
        for i in pbar:
            prompt, weight = next(prompt_gen)
            # get text embeddings for unconditional and conditional prompts
            emb_0 = theta_o.get_learned_conditioning([""])
            emb_p = theta_o.get_learned_conditioning([prompt])
            emb_n = theta_u.get_learned_conditioning([prompt])

            opt.zero_grad()

            t_enc = torch.randint(self.ddim_steps, (1,), device=theta_u.device)
            # time step from 1000 to 0 (0 being good)
            og_num = round((int(t_enc) / self.ddim_steps) * 1000)
            og_num_lim = round((int(t_enc + 1) / self.ddim_steps) * 1000)

            t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=theta_u.device)

            start_code = torch.randn((1, 4, 64, 64)).to(theta_u.device)

            with torch.no_grad():
                z = quick_sample_till_t(
                    emb_p.to(theta_u.device),
                    self.start_guidance,
                    start_code,
                    int(t_enc),
                )
                e_0 = theta_o.apply_model(
                    z.to(theta_o.device),
                    t_enc_ddpm.to(theta_o.device),
                    emb_0.to(theta_o.device),
                )
                e_p = theta_o.apply_model(
                    z.to(theta_o.device),
                    t_enc_ddpm.to(theta_o.device),
                    emb_p.to(theta_o.device),
                )
            e_n = theta_u.apply_model(
                z.to(theta_u.device),
                t_enc_ddpm.to(theta_u.device),
                emb_n.to(theta_u.device),
            )
            e_0.requires_grad = False
            e_p.requires_grad = False

            loss = weight.to(theta_u.device) * criteria(
                e_n.to(theta_u.device),
                e_0.to(theta_u.device)
                - (
                    self.negative_guidance
                    * (e_p.to(theta_u.device) - e_0.to(theta_u.device))
                ),
            )

            loss.backward()
            if self.wandb is not None:
                self.wandb.log({"loss": loss.item()})
            pbar.set_postfix({"loss": loss.item()})
            opt.step()
        theta_u.eval()

        os.makedirs(self.output_name, exist_ok=True)
        torch.save({"state_dict": theta_u.state_dict()}, os.path.join(self.output_name, "model.pth"))

        # visualize 
        num_inference_steps = 50
        image_size = 512

        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        for prompt in prompts:
            
            all_images = [] 
            prompt_dir = os.path.join(self.output_name, f'prompt_{prompt}')
            os.makedirs(prompt_dir, exist_ok=True)

            for _ in range(20):
                # https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/scripts/txt2img.py#L289
                uncond_embeddings = theta_u.get_learned_conditioning(1 * [""])
                text_embeddings = theta_u.get_learned_conditioning(1 * [prompt])
                print(str(prompt))
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            
                height = image_size
                width = image_size
            
                latents = torch.randn(
                    (1, 4, height // 8, width // 8)
                )
                latents = latents.to(theta_u.device)
            
                scheduler.set_timesteps(num_inference_steps)
            
                latents = latents * scheduler.init_noise_sigma      
                scheduler.set_timesteps(num_inference_steps)
            
                for t in tqdm(scheduler.timesteps):
                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latents] * 2)
            
                    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
            
                    # predict the noise residual
                    with torch.no_grad():
                        t_unet = torch.full((1, ), t, device=theta_u.device)
                        noise_pred = theta_u.apply_model(latent_model_input, t_unet, text_embeddings)
            
                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            
                    # compute the previous noisy sample x_t -> x_t-1
                    latents = scheduler.step(noise_pred, t, latents).prev_sample
                
                with torch.no_grad():
                    image = theta_u.decode_first_stage(latents)
            
                image = (image / 2 + 0.5).clamp(0, 1)
                all_images.append(image)

                for j, image in enumerate(all_images):
                    # Save each image separately
                    image_path = os.path.join(prompt_dir, f'image_{j}.jpg')
                    image = (image / 2 + 0.5).clamp(0, 1).squeeze(0)
                    grid = 255. * rearrange(image, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img.save(image_path)
            
        return theta_u


class Eval:
    def __init__(self, eval_loader, ddim_steps=50):
        self.eval_loader = eval_loader
        self.ddim_steps = ddim_steps

    @torch.no_grad()
    def eval(self, theta_u, prompts):
        print("Eval prompts:", prompts)

        theta_u.eval()

        w_grad = torch.zeros(len(prompts))

        criteria = torch.nn.MSELoss(reduction="none")

        pbar = tqdm(self.eval_loader)

        p = theta_u.num_timesteps // 10
        for images, prompts, idxs in pbar:

            for t in range(0, theta_u.num_timesteps, p):
                batch = {
                    theta_u.first_stage_key: images.to(theta_u.device),
                    theta_u.cond_stage_key: {"c_crossattn": prompts[0]},
                }

                input, emb = theta_u.get_input(batch, theta_u.first_stage_key)
                noise = torch.randn_like(input, device=theta_u.device)

                tensor_t = torch.full((input.shape[0],), t, device=theta_u.device, dtype=torch.int64)
                noisy = theta_u.q_sample(x_start=input, t=tensor_t, noise=noise)

                out = theta_u.apply_model(noisy, tensor_t, emb)
                loss = criteria(out, noise / theta_u.num_timesteps).mean(dim=[1,2,3])
                pbar.set_postfix({"loss": loss.mean().item()})
                w_grad[idxs] += loss.to("cpu")

        return w_grad / len(self.eval_loader.dataset) / 10

class SignSGD(torch.optim.SGD):
    def __init__(self, params, lr, momentum, weight_decay):
        super().__init__(params, lr, momentum, weight_decay)

    def sign_step(self):
        """Performs a single optimization step using the sign of gradients."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.sign()
                p.data.add_(d_p, alpha=-group["lr"])


class SignESD(ESD):
    def get_optimizer(self, parameters):
        return SignSGD(
            parameters,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def __init__(
        self,
        train_method,
        start_guidance,
        negative_guidance,
        iterations,
        eval_iters,
        lr,
        momentum,
        weight_decay,
        output_name,
        image_size=512,
        ddim_steps=50,
        wandb=wandb,
    ):
        self.momentum = momentum
        self.weight_decay = weight_decay
        super().__init__(
            train_method=train_method,
            start_guidance=start_guidance,
            negative_guidance=negative_guidance,
            iterations=iterations,
            eval_iters=eval_iters,
            lr=lr,
            output_name=output_name,
            image_size=image_size,
            ddim_steps=ddim_steps,
            wandb=wandb,
        )