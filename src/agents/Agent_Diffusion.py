import numpy as np
from src.agents.Agent_NCA import Agent_NCA
from src.agents.Agent_Multi_NCA import Agent_Multi_NCA
import torch
import random
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
import os
from src.agents.Agent_Multi_NCA import EMA

class Agent_Diffusion(Agent_Multi_NCA):
    def initialize(self, beta_schedule='linear'): 
        super().initialize()
        self.timesteps = self.exp.get_from_config('timesteps')
        self.beta_schedule = self.exp.get_from_config('schedule') 
        self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, self.betas, self.sqrt_recip_alphas, \
            self.posterior_variance = self.calc_schedule()
        self.averages = False
        self.ema = EMA(self.model[0], decay=0.99)

    @staticmethod
    def extract(a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def gaussian_kernel(self, size, sigma):
        """Creates a 2D Gaussian kernel."""
        x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.double).to(self.device)
        y = x.unsqueeze(1)
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)  
    def apply_gaussian_blur(self, images, t_values, max_kernel_size=17):
        blurred_images = []

        t_values = t_values / self.timesteps  

        for i, image in enumerate(images):
            # Adjust sigma based on t value, starting from 0
            sigma = t_values[i] * 3  # Adjust the multiplier as needed for maximum blur


            # Skip blurring if sigma is effectively 0
            if sigma < 0.01:
                blurred_images.append(image)
                continue

            # Ensure kernel size is odd and within bounds
            kernel_size = min(max_kernel_size, int(1 + 2 * torch.ceil(sigma * 3)))
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

            kernel = self.gaussian_kernel(kernel_size, sigma)
            kernel = kernel.expand(3, 1, kernel_size, kernel_size)  # Assuming RGB images

            # Apply the Gaussian kernel with padding to maintain image size
            padding = kernel_size // 2
            blurred_image = F.conv2d(image.unsqueeze(0), kernel, padding=padding, groups=3)
            blurred_images.append(blurred_image[0])

        return torch.stack(blurred_images)
    
    def q_sample(self, x_start, t, noise=None):
        
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        noisy_image = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

        return noisy_image

    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = (torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2) 
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    @staticmethod
    def linear_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    @staticmethod
    def quadratic_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2

    @staticmethod
    def sigmoid_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    def calc_schedule_wrong(self):
        betas = torch.linspace(0, 1, self.timesteps).to(self.device)
        alphas = 1 - betas
        sqrt_alphas_cumprod = alphas
        sqrt_one_minus_alphas_cumprod = betas

        posterior_variance = betas 
        sqrt_recip_alphas = alphas

        return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, betas, sqrt_recip_alphas, posterior_variance
    
    def calc_schedule(self):
        # define beta schedule
        betas = 0
        if self.beta_schedule == "linear":
            betas = self.linear_beta_schedule(timesteps=self.timesteps)
        elif self.beta_schedule == "cosine":
            betas = self.cosine_beta_schedule(timesteps=self.timesteps)
        elif self.beta_schedule == "quadratic":
            betas = self.quadratic_beta_schedule(timesteps=self.timesteps)
        elif self.beta_schedule == "sigmoid":
            betas = self.sigmoid_beta_schedule(timesteps=self.timesteps)
        else:
            NotImplementedError()

        betas = betas.to(self.device)

        # define alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.device)  # (alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0).to(self.device)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, betas, sqrt_recip_alphas, posterior_variance

    def prepare_data(self, data, t, label=None, eval=False):
        r"""
        preprocessing of data
        :param data: images
        :param t: current time steps
        :param batch_size:
        :param label:
        :return: corrupt images, associated noise
        """
        id, img, _ = data
        img = img.to(self.device)
        
        noise, img_noisy = self.getNoiseLike(img, noisy=True, t=t)
        
        img_noisy = self.make_seed(img_noisy)
        if not eval:
            img_noisy, noise = self.repeatBatch(img_noisy, noise, self.exp.get_from_config('batch_duplication'))
        data_noisy = (id, img_noisy, img_noisy)



        return data_noisy, noise, label

    def get_outputs(self, data, full_img=False, t=0, mask=None, **kwargs):
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        t = torch.tensor(t/self.timesteps).to(self.device) 
        id, inputs, targets = data
        if self.exp.model_state == "train":
            t = t.repeat(self.exp.get_from_config('batch_duplication'))
        
        if isinstance(self.model, list): # TODO: Add support for > batch size one
            if torch.numel(t) > 1:
                model_id = math.floor(((t[0]-0.0000001) * self.timesteps) / (self.timesteps / len(self.model)))   
            else:
                model_id = math.floor(((t-0.0000001) * self.timesteps) / (self.timesteps / len(self.model)))  
            outputs = self.model[model_id](inputs, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'), t=t, epoch=self.exp.currentStep, mask=mask)       

        else:
            outputs = self.model(inputs, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'), t=t, epoch=self.exp.currentStep, mask=mask)
        
        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach().cpu(), id)
        return outputs[..., self.input_channels:self.input_channels+self.output_channels], targets

    def rescale_image(self, data):
        id, img, label = data

        random_fac = random.uniform(0.5,1)

        img = img.transpose(1,3)
        size = (int(img.shape[2]*random_fac), int(img.shape[3]*random_fac)) 

        img = F.interpolate(img, size=size, mode='bilinear')
        img = img.transpose(1,3)

        return id, img, label

    def batch_step(self, data):
        r"""Execute a single batch training step
            #Args
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            #Returns:
                loss item
        """
        if isinstance(self.model, list):
            rang = int((self.timesteps / len(self.model)) * np.random.randint(0, len(self.model)))
            t = torch.randint(0, int(self.timesteps / len(self.model)), (data[1].shape[0],), device=self.exp.get_from_config(tag="device")).long()
            t = torch.add(t, rang)
        else:
            t = torch.randint(0, self.timesteps, (data[1].shape[0],), device=self.exp.get_from_config(tag="device")).long()
        
        data, noise, _ = self.prepare_data(data, t)
        outputs, _ = self.get_outputs(data, t=t)

        if isinstance(self.optimizer, list): 
            for m in range(len(self.optimizer)):
                self.optimizer[m].zero_grad()
        else:
            self.optimizer.zero_grad()
        loss_ret = {}
        loss = F.mse_loss(outputs, noise) + F.l1_loss(outputs, noise)

        loss_ret[0] = loss
        loss.backward()
        if isinstance(self.optimizer, list): 
            for m in range(len(self.optimizer)):
                self.optimizer[m].step()
        else:
            self.optimizer.step()
        if isinstance(self.scheduler, list): 
            for m in range(len(self.scheduler)):
                self.scheduler[m].step()
        else:
            self.scheduler.step()
            

        self.ema.update()

        
        return loss_ret

    def getNoiseLike(self, img, noisy=False, t=0):

        def getNoise():
            rnd = torch.randn_like(img).to(self.device).to(torch.float) 
            return rnd 
        if noisy:
            noise = getNoise()
            img_noisy = self.q_sample(x_start=img, t=t, noise=noise)
            img_noisy = img_noisy.to(self.device)
            noise = noise
        else:
            noise = getNoise()
            img_noisy = 0

        return noise.to(self.device), img_noisy

    @torch.no_grad()
    def p_sample(self, output, x, t, t_index):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * output / sqrt_one_minus_alphas_cumprod_t
        )
        # return output
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise, _ = self.getNoiseLike(x) 
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def intermediate_evaluation(self, dataloader, epoch):
        self.exp.set_model_state("test")
        if epoch % 5000 == 0:
            self.test_fid()
        self.test()
        self.exp.set_model_state("train")

    def generateSamples(self, samples:int=1, normal=True):
        r"""Get the average Dice test score.
            #Returns:
                return (float): Average Dice score of test set. """
        self.exp.set_model_state("test")
        self.test(tag="extra", samples=samples, extra=True, normal=normal)

    @torch.no_grad()
    def test_fid(self, tag:str='0', samples:int=1, extra:bool=False, optimized=False, saveImg=False, scale=1.0, inp_id = None, **kwargs):
        size = self.exp.get_from_config('input_size')

        self.ema.apply_shadow()

        tag = ""

        if samples < 2: samples = 2

        # Generate samples
        noise, _ = self.getNoiseLike(torch.zeros((samples, int(size[0]*scale), int(size[1]*scale), self.exp.get_from_config('input_channels'))))
        
        if scale != 1.0:
            print("Scale = ", int(size[0]*scale), int(size[1]*scale))
            tag = "_" + str(scale)
        img = self.make_seed(noise)
            
        for step in tqdm(reversed(range(self.timesteps))):
            t = torch.full((samples,), step, device=self.device, dtype=torch.long)
            img_p = 0, img, 0
            output, _ = self.get_outputs(img_p, t = step)
            img = self.p_sample(output, img[...,0:self.exp.get_from_config('input_channels')], t, step)
            img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])

        # SAVE IMAGES
        imgs = (img.detach().cpu().numpy()+1)/2
        if saveImg and not optimized:
            for i, b in enumerate(range(img.shape[0])):
                # SAVE IMAGE
                name = random.randint(0, 2000000000)
                path = os.path.join(self.exp.get_from_config('model_path'), "models", "epoch_" + str(self.exp.currentStep-1), "Generated" + tag)
                if not os.path.exists(path):
                    os.makedirs(path)
                path = os.path.join(path, str(name)+ ".jpg")
                cv2.imwrite(path, cv2.cvtColor(np.clip(imgs[b, ..., 0:self.exp.get_from_config('input_channels')]*255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


        # Compose images
        factor = 4
        composition = np.zeros((factor*size[0], factor*size[1], self.exp.get_from_config('input_channels')))
        if size[0]*scale == 64:
            for i, b in enumerate(range(img.shape[0])):
                x = b % 4
                y = int(math.floor(b//factor))
                composition[x*size[0]:(x+1)*size[0], y*size[0]:(y+1)*size[0], 0:self.exp.get_from_config('input_channels')] = imgs[b, ... , 0:self.exp.get_from_config('input_channels')] 
                if i == (factor*factor -1): break
            self.exp.write_img("Composition", composition, self.exp.currentStep, normalize=True) 

        else:
            return

        sample = (((img[..., 0:self.exp.get_from_config('input_channels')].transpose(1,3).detach().cpu()+1)/2)*256) #
        sample = torch.clip(sample, 0, 255).to(torch.uint8)
        self.exp.getFID().update(sample, real=False)
        fid_score = self.exp.fid.compute()
        print("FID: ", fid_score)
        self.exp.write_scalar('FID', fid_score, self.exp.currentStep)

        self.ema.restore_original()

    def calculateFID_fromFiles(self, samples, scale = 1.0):
        imgs = None

        tag = ""
        size = self.exp.get_from_config('input_size')
        if scale != 1.0:
            print("Scale = ", int(size[0]*scale), int(size[1]*scale))
            tag = "_" + str(scale)

        path = os.path.join(self.exp.get_from_config('model_path'), "models", "epoch_" + str(self.exp.currentStep-1), "Generated" + tag)

        for i, file in tqdm(enumerate(os.listdir(path)), total=len(os.listdir(path))):
            img = torch.unsqueeze(torch.tensor(cv2.cvtColor(cv2.imread(os.path.join(path, file)), cv2.COLOR_BGR2RGB)).to(torch.uint8), dim=0)

            if imgs is None:
                imgs = img 
            else:
                imgs = torch.cat((imgs, img), dim=0)
            # Only use 2048 samples
            if i == 2047: break

        self.exp.getKID().update(imgs.transpose(1,3), real=False)
        print("KID: ", self.exp.kid.compute())
        self.exp.getFID().update(imgs.transpose(1,3), real=False)
        print("FID: ", self.exp.fid.compute())
        imgs = None

    @torch.no_grad()
    def test(self, tag='0', samples=1, extra=False, normal=True, **kwargs):

        batch_size = 1
        # Generate sample
        size = self.exp.get_from_config('input_size')
        
        dataset = self.exp.dataset
        self.exp.set_model_state('test')

        self.ema.apply_shadow()
        self.timesteps = 300



        if normal:
            for s in range(1):
                self.ema.apply_shadow()
                noise, _ = self.getNoiseLike(torch.zeros((1, size[0], size[1], self.exp.get_from_config('input_channels'))))
                torch.cuda.reset_max_memory_allocated()
                currentmax = torch.cuda.max_memory_allocated()

                img = self.make_seed(noise)
                for step in tqdm(reversed(range(self.timesteps))):
                    t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
                    img_p = 0, img, 0
                    output, _ = self.get_outputs(img_p, t = step)
                    img = self.p_sample(output, img[...,0:self.exp.get_from_config('input_channels')], t, step)
                    img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])
                
                max_memory = torch.cuda.max_memory_allocated()  # in bytes
                print(f"Maximum VRAM used: {(max_memory-currentmax) / 1024 ** 2} MB")
                
                plt.figure(figsize=(6, 6))
                plt.imshow((img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy()+1)/2)
                plt.axis('off')
                plt.show()

                self.exp.write_img(tag, (img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy()+1)/2, self.exp.currentStep, context={'Image':s}, normalize=True) #/2+0.5 #{'Image':s}
                self.ema.restore_original()